import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from typing import List, Tuple

# --- Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- JIT-Compiled Dynamics & Loss Functions ---
@torch.jit.script
def V_step_jit(V: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    phi, psi, dt = F[0], F[1], F[2]
    dt2 = dt * dt / 2
    x, y, a, s, w, t = V[0], V[1], V[2], V[3], V[4], V[5]
    _0 = torch.zeros_like(x)
    _1 = torch.ones_like(x)
    A1 = torch.stack([x, y, a, s, w, t])
    A2 = torch.stack([s*torch.cos(a), s*torch.sin(a), w, _0, _0, _0])
    A3 = torch.stack([-s*w*torch.sin(a), s*w*torch.cos(a), _0, _0, _0, _0])
    A = A1 + dt*A2 + dt2*A3
    B2 = torch.stack([_0, _0, _0, _1, _0, _0])
    B3 = torch.stack([torch.cos(a), torch.sin(a), _0, _0, _0, _0])
    B = dt*B2 + dt2*B3
    C2 = torch.stack([_0, _0, _0, _0, _1, _0])
    C3 = torch.stack([_0, _0, _1, _0, _0, _0])
    C = dt*C2 + dt2*C3
    D = torch.stack([_0, _0, _0, _0, _0, _1])
    return A + phi*B + psi*C + dt*D

@torch.jit.script
def loss_fun_jit(V_end: torch.Tensor, target: torch.Tensor, F: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    diff = V_end[:5] - target
    target_loss = diff.dot(diff)
    time_loss = V_end[5]
    l2_loss = (F[:,:2]**2).mean()
    return target_loss, time_loss, l2_loss

@torch.jit.script
def get_forcing_jit(V: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    S = V[3].abs()
    phi = F[0].clamp(-0.8, 0.8)
    psi_lim = torch.min(S, torch.tensor(0.8, device=F.device, dtype=F.dtype))
    psi = F[1].clamp(-psi_lim, psi_lim)
    dt = F[2].clamp(0.001, 0.25)
    return torch.stack([phi, psi, dt])

@torch.jit.script
def compute_path_autograd_jit(
    v0: torch.Tensor,
    F_req: torch.Tensor,
    target: torch.Tensor
) -> Tuple[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    V_list: List[torch.Tensor] = []
    V_list.append(v0)
    F_true: List[torch.Tensor] = []
    for i in range(F_req.size(0)):
        V_prev = V_list[i]
        F_clamped = get_forcing_jit(V_prev, F_req[i])
        F_true.append(F_clamped)
        V_list.append(V_step_jit(V_prev, F_clamped))
    V_end = V_list[-1]
    F_true_stack = torch.stack(F_true)
    Lt, Tt, Ll = loss_fun_jit(V_end, target, F_true_stack)
    return V_list, (Lt, Tt, Ll)

# --- Batch Generation of Examples ---
def gen_batch(
    v0: torch.Tensor,
    target: torch.Tensor,
    n_steps: int,
    batch_size: int,
    max_iter: int = 10
) -> torch.Tensor:
    """Generate a batch of optimized forcing sequences."""
    F = torch.zeros(batch_size, n_steps, 3, device=device, requires_grad=True)
    optimizer_F = optim.AdamW([F], lr=6e-2)
    scaler = torch.amp.GradScaler()
    for _ in range(max_iter):
        optimizer_F.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        # accumulate loss over batch
        for b in range(batch_size):
            with torch.amp.autocast():
                _, (Lt, Tt, Ll) = compute_path_autograd_jit(v0, F[b], target)
                total_loss = total_loss + (Lt + 1e-8*Tt - 1.0*Ll)
        scaler.scale(total_loss).backward()
        scaler.step(optimizer_F)
        scaler.update()
        # clamp controls
        with torch.no_grad():
            for b in range(batch_size):
                V_list, _ = compute_path_autograd_jit(v0, F[b], target)
                for i in range(n_steps):
                    F.data[b, i] = get_forcing_jit(V_list[i], F.data[b, i])
    return F.detach()

# --- RNN Model Definition ---
class TrajectoryRNN(nn.Module):
    def __init__(self, state_dim=6, ctrl_dim=3, hidden_dim=10, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(
            state_dim + ctrl_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.1
        )
        self.fc = nn.Linear(hidden_dim, ctrl_dim)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, states: torch.Tensor, ctrls: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(torch.cat([states, ctrls], dim=-1))
        return self.fc(out)

# --- Main Execution ---
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    # Time grid settings
    t0, t1, dt_max = 0.0, 5.5, 0.25
    n_steps = int(2 ** np.ceil((t1 - t0) / dt_max))
    print(f"n_steps = {n_steps}")

    # Initial conditions and target
    v0 = torch.tensor([0.0, 0.0, 0.2, 1.0, 1.0, t0], device=device)
    target = torch.tensor(
        [5.68071168, 2.5029068, -1.8625, 2.375, -1.75],
        device=device
    )

    # Generate training examples in batches
    num_ex = 64
    batch_size = 8
    print("Generating examples in batches...")
    examples = []
    for _ in range(num_ex // batch_size):
        F_batch = gen_batch(v0, target, n_steps, batch_size, max_iter=10)
        examples.extend([F_batch[b] for b in range(batch_size)])

    # DataLoader
    class ExDataset(Dataset):
        def __init__(self, Fs): self.Fs = Fs
        def __len__(self): return len(self.Fs)
        def __getitem__(self, i): return self.Fs[i]

    loader = DataLoader(
        ExDataset(examples),
        batch_size=batch_size,
        shuffle=True
    )

    # Instantiate and train RNN
    model = TrajectoryRNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    epochs = 50
    print("Training RNN...")
    scaler = torch.amp.GradScaler()
    for ep in range(epochs):
        total_loss = 0.0
        for F_gt in loader:
            B = F_gt.size(0)
            states = torch.zeros(B, n_steps, 6, device=device)
            ctrls = torch.zeros(B, n_steps, 3, device=device)
            optimizer.zero_grad()
            with torch.amp.autocast():
                pred = model(states, ctrls)
                loss = (F_gt - pred).abs().mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {ep+1}/{epochs}, MAE={total_loss/len(loader):.4f}")

    # One-shot inference + refinement
    states = torch.zeros(1, n_steps, 6, device=device)
    ctrls = torch.zeros(1, n_steps, 3, device=device)
    t_start = time.perf_counter()
    with torch.no_grad():
        F0 = model(states, ctrls).squeeze(0)
    infer_ms = (time.perf_counter() - t_start) * 1000
    print(f"Inference time: {infer_ms:.2f} ms")

    F_ref = F0.clone().requires_grad_(True)
    opt_ref = optim.AdamW([F_ref], lr=6e-2)
    print("Refining output...")
    for _ in range(10):
        opt_ref.zero_grad()
        with autocast():
            _, (Lt, Tt, Ll) = compute_path_autograd_jit(v0, F_ref, target)
            L = Lt + 1e-8 * Tt - 1.0 * Ll
        scaler.scale(L).backward()
        scaler.step(opt_ref)
        scaler.update()
        with torch.no_grad():
            V_list, _ = compute_path_autograd_jit(v0, F_ref, target)
            for i in range(n_steps):
                F_ref.data[i] = get_forcing_jit(V_list[i], F_ref.data[i])

    # Plot ΔF
    Fnp = F_ref.cpu().numpy()
    ph = Fnp[:, 0]; ps = Fnp[:, 1]; dt_vals = Fnp[:, 2]
    times = np.cumsum(dt_vals)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(times, ph, label='phi')
    plt.plot(times, ps, label='psi')
    plt.legend()
    plt.title('Acceleration Profile')

    plt.subplot(1, 2, 2)
    plt.plot(times, dt_vals)
    plt.title('Δt Sequence')

    plt.tight_layout()
    plt.show()
