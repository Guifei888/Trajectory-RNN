import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- RNN that predicts delta forcing (ΔF) over a sequence ---
class ForcingRNN(nn.Module):
    def __init__(self, state_dim=5, control_dim=2, hidden_size=64, num_layers=1):
        super(ForcingRNN, self).__init__()
        self.rnn = nn.GRU(input_size=state_dim + control_dim + state_dim,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, control_dim)

    def forward(self, V_seq, F0_seq, target_seq):
        # V_seq: [B, T, state_dim], F0_seq: [B, T, C], target_seq: [B, T, state_dim]
        inp = torch.cat([V_seq, F0_seq, target_seq], dim=-1)
        out, _ = self.rnn(inp)
        delta = self.fc(out)
        # Delta-prediction + warm-start
        return F0_seq + delta

# --- Dynamics: include both φ (accel) and ψ (turn) controls ---
def v_prime_components(v, dt):
    dt2 = dt * dt / 2
    x, y, a, s, w = v.unbind(-1)
    # drift terms (Taylor to second order)
    A1 = torch.stack([x, y, a, s, w], dim=-1)
    A2 = torch.stack([s*torch.cos(a), s*torch.sin(a), w, 0, 0], dim=-1)
    A3 = torch.stack([-s*w*torch.sin(a), s*w*torch.cos(a), 0, 0, 0], dim=-1)
    A = A1 + dt * A2 + dt2 * A3
    # control bases
    B_phi = dt * torch.tensor([0, 0, 0, 1, 0], device=v.device, dtype=v.dtype)
    B_psi = dt * torch.tensor([0, 0, 0, 0, 1], device=v.device, dtype=v.dtype)
    return A, B_phi, B_psi

# --- Simulate under a forcing sequence ---
def simulate(v0, F_seq, ts):
    V = torch.zeros((len(ts), 5), device=v0.device, dtype=v0.dtype)
    V[0] = v0
    for i in range(len(ts) - 1):
        dt = ts[i+1] - ts[i]
        A, B_phi, B_psi = v_prime_components(V[i], dt)
        phi, psi = F_seq[i]
        V[i+1] = A + phi * B_phi + psi * B_psi
    return V

# --- Small-Adam polish on forcing sequence ---
def refine_forcing(v0, F_init, ts, target_seq, n_steps=10, lr=1e-2):
    F = F_init.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([F], lr=lr)
    criterion = nn.MSELoss()
    for _ in range(n_steps):
        optimizer.zero_grad()
        V = simulate(v0, F, ts)
        loss = criterion(V, target_seq)
        loss.backward()
        optimizer.step()
    return F.detach(), loss.item()

# --- Meta-loss: count steps needed to converge under inner Adam ---
def compute_meta_loss(v0, F_pred, ts, target_seq,
                      max_steps=20, tol=1e-2, inner_lr=1e-2):
    F = F_pred.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([F], lr=inner_lr)
    criterion = nn.MSELoss()
    initial_loss = criterion(simulate(v0, F, ts), target_seq).item()
    steps_needed = max_steps
    for i in range(max_steps):
        optimizer.zero_grad()
        V = simulate(v0, F, ts)
        loss = criterion(V, target_seq)
        if loss.item() < tol * initial_loss:
            steps_needed = i
            break
        loss.backward()
        optimizer.step()
    final_loss = loss.item()
    # penalize fraction of max_steps used
    meta = final_loss + (steps_needed / max_steps)
    return meta

# --- Training loop with primary + meta loss ---
def train_rnn_with_meta_loss(model, data_loader, ts, epochs=50,
                             lr=1e-3, lambda_meta=0.1, device='cpu'):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for ep in range(epochs):
        total_loss = 0.0
        for v0_batch, F_gt_batch, target_seq_batch in data_loader:
            # Ensure float32
            v0_batch = v0_batch.float().to(device)           # [B,5]
            F_gt_batch = F_gt_batch.to(device)              # [B,T,2]
            target_seq_batch = target_seq_batch.to(device)  # [B,T,5]

            B = v0_batch.size(0)
            T = ts.size(0)
            # Prepare input V_seq: tile initial state across time
            V_seq = v0_batch.unsqueeze(1).repeat(1, T, 1)   # [B,T,5]
            # Warm start forcing
            F0 = torch.zeros_like(F_gt_batch)

            # RNN prediction
            F_pred = model(V_seq, F0, target_seq_batch)

            # Primary loss on ΔF match
            loss_main = criterion(F_pred, F_gt_batch)
            # Meta loss on polish difficulty (first sample only for speed)
            meta = compute_meta_loss(v0_batch[0], F_pred[0], ts, target_seq_batch[0])
            loss = loss_main + lambda_meta * meta

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {ep+1}/{epochs} avg_loss={avg_loss:.4f}")

# --- Plotting utility ---
def showTraj(V, ax, label=None, **kwargs):
    x = V[:,0].cpu().numpy()
    y = V[:,1].cpu().numpy()
    ax.plot(x, y, label=label, **kwargs)
    if label:
        ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', 'box')

# --- Example main script ---
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    T = 50
    ts = torch.linspace(0, 5, T, dtype=torch.float32, device=device)

    # Dummy dataset: random start/end positions
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, N):
            self.data = []
            for _ in range(N):
                x0, y0 = np.random.uniform(-5,5,2)
                xT, yT = np.random.uniform(-5,5,2)
                # Ensure float32 construction
                v0 = torch.tensor([x0, y0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
                # ground truth: straight-line forcing
                dt_total = ts[-1].item()
                F_phi = (xT - x0) / dt_total
                F_psi = (yT - y0) / dt_total
                F_gt = torch.stack([
                    torch.full((T,), F_phi, dtype=torch.float32, device=device),
                    torch.full((T,), F_psi, dtype=torch.float32, device=device)
                ], dim=-1)
                # target_seq: replicate end state for all t for simplicity
                target_seq = torch.zeros((T,5), dtype=torch.float32, device=device)
                target_seq[-1,0] = xT
                target_seq[-1,1] = yT
                self.data.append((v0, F_gt, target_seq))
        def __len__(self): return len(self.data)
        def __getitem__(self, idx): return self.data[idx]

    dataset = DummyDataset(100)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    # Build and train
    model = ForcingRNN()
    train_rnn_with_meta_loss(model, loader, ts, epochs=20, device=device)

    # Pick one sample for visualization
    v0, F_gt, target_seq = dataset[0]
    # Prepare V_seq for single sample
    V_seq_single = v0.unsqueeze(0).unsqueeze(1).repeat(1, T, 1)  # [1,T,5]
    F0_single = torch.zeros_like(F_gt, dtype=torch.float32).unsqueeze(0)             # [1,T,2]
    target_single = target_seq.unsqueeze(0)                     # [1,T,5]
    F_rnn = model(V_seq_single, F0_single, target_single).squeeze(0)
    F_refined, _ = refine_forcing(v0, F_rnn, ts, target_seq)

    # Simulate trajectories
    V_gt = simulate(v0, F_gt, ts)
    V_rnn = simulate(v0, F_rnn, ts)
    V_ref = simulate(v0, F_refined, ts)

    # Plot
    fig, ax = plt.subplots()
    showTraj(V_gt,  ax, label='Ground Truth',    linestyle='-')
    showTraj(V_rnn, ax, label='RNN Warm Start',   linestyle='--')
    showTraj(V_ref, ax, label='RNN + Adam Refine', linestyle='-.')
    ax.scatter(V_gt[0,0], V_gt[0,1], marker='o', s=50)
    ax.scatter(V_gt[-1,0],V_gt[-1,1],marker='X', s=50)
    plt.show()

if __name__ == "__main__":
    main()
