# trajectory_model.py
# Integrated LSTM/GRU trajectory & forcing solver with data generation, logging, and main execution

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Visualization helper ---
def showTraj(V_mat, F_mat, ts, target, figaxs=None, show=True, alpha=0.5):
    if figaxs is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=200)
    else:
        fig, (ax1, ax2) = figaxs
    ax1.plot(V_mat[:,0], V_mat[:,1], '-o', alpha=alpha)
    ax1.scatter([target[0]], [target[1]], c='r', label='Target')
    ax1.set_title('Position')
    ax1.set_aspect('equal', 'box')
    ax1.legend()
    ax2.plot(ts[:-1], F_mat[:,0], label='ϕ')
    ax2.plot(ts[:-1], F_mat[:,1], label='ψ')
    ax2.set_title('Forcing inputs')
    ax2.set_xlabel('Time')
    ax2.legend()
    if show:
        plt.show()
    else:
        return fig, (ax1, ax2)

# --- Dynamics integration (second-order Taylor) ---
def v_prime_components(v, dt):
    dt2 = dt * dt / 2
    x, y, a, s, w = v
    zero = torch.zeros_like(a)
    one = torch.ones_like(a)
    A1 = torch.stack([x, y, a, s, w])
    A2 = torch.stack([s * torch.cos(a), s * torch.sin(a), w, zero, zero])
    A3 = torch.stack([-s * w * torch.sin(a), s * w * torch.cos(a), zero, zero, zero])
    A = A1 + dt * A2 + dt2 * A3
    B_base = torch.stack([zero, zero, zero, one, zero])
    B_force = torch.stack([torch.cos(a), torch.sin(a), zero, zero, zero])
    C_base = torch.stack([zero, zero, zero, zero, one])
    C_force = torch.stack([zero, zero, one, zero, zero])
    B = dt * B_base + dt2 * B_force
    C = dt * C_base + dt2 * C_force
    return A, B, C

# --- Path gradient solver ---
def compute_path_gradient(v0, F_mat, ts, target):
    n = len(ts)
    V_mat = torch.zeros((n,5), device=device)
    V_mat[0] = v0
    for i in range(n-1):
        A, B, C = v_prime_components(V_mat[i], ts[i+1]-ts[i])
        V_mat[i+1] = A + F_mat[i,0]*B + F_mat[i,1]*C
    loss = nn.MSELoss()(V_mat[-1], target)
    jacobians = [None]*n
    jacobians[-1] = torch.eye(5, device=device)
    for i in range(n-2, -1, -1):
        V_i = V_mat[i].detach().clone().requires_grad_(True)
        _, B, C = v_prime_components(V_i, ts[i+1]-ts[i])
        def next_state(v):
            A0, B0, C0 = v_prime_components(v, ts[i+1]-ts[i])
            return A0 + F_mat[i,0]*B0 + F_mat[i,1]*C0
        J = torch.autograd.functional.jacobian(next_state, V_i)
        jacobians[i] = jacobians[i+1] @ J
    YT = (2/5)*(V_mat[-1] - target)
    grad = torch.zeros_like(F_mat)
    for j in range(n-1):
        _, B, C = v_prime_components(V_mat[j], ts[j+1]-ts[j])
        contrib = jacobians[j+1] @ torch.stack([B, C], dim=1)
        grad[j] = YT @ contrib
    return V_mat, grad, loss

# --- Gradient-based forcing solver ---
def solveForcingWithGrad(v0=None, target=None, t0=0, t1=5.5, dt_max=0.25,
                          lr=6e-2, weight_decay=1e-1,
                          nsteps=1000, break_thresh=1e-5):
    if v0 is None:
        v0 = torch.tensor([0.,0.,0.2,1.,1.], device=device)
    if target is None:
        target = torch.tensor([5.68,2.50,-1.86,2.375,-1.75], device=device)
    n_ts = int((t1-t0)/dt_max) + 1
    ts = torch.linspace(t0, t1, n_ts, device=device)
    F_mat = nn.Parameter(torch.randn((n_ts-1,2), device=device).clamp(-0.8,0.8))
    optimizer = optim.AdamW([F_mat], lr=lr, weight_decay=weight_decay)
    start = time.time()
    for i in range(nsteps):
        V_mat, grad, loss = compute_path_gradient(v0, F_mat, ts, target)
        optimizer.zero_grad()
        F_mat.grad = grad
        optimizer.step()
        F_mat.data.clamp_(-0.8,0.8)
        if (i+1) % 100 == 0:
            print(f"Iter {i+1}/{nsteps} | RMSE={loss.item()**0.5:.4f} | Elapsed {time.time()-start:.1f}s", flush=True)
        if loss.item()**0.5 < break_thresh:
            break
    print(f"Solver done in {time.time()-start:.1f}s | Final RMSE={loss.item()**0.5:.4f}", flush=True)
    showTraj(V_mat.detach().cpu().numpy(), F_mat.detach().cpu().numpy(), ts.cpu().numpy(), target.cpu().numpy())
    return F_mat.detach(), ts.detach()

# --- Data generation for RNN training ---
def generate_training_data(n_samples, t0=0, t1=5.5, dt_max=0.25,
                            nsteps=200, lr=6e-2, weight_decay=1e-1,
                            break_thresh=1e-5):
    sequences, targets = [], []
    for i in range(n_samples):
        print(f"Generating sample {i+1}/{n_samples}...", flush=True)
        v0 = torch.tensor([np.random.uniform(-5,5), np.random.uniform(-5,5),
                            np.random.uniform(-np.pi,np.pi), np.random.uniform(0,5),
                            np.random.uniform(-2,2)], device=device)
        target = torch.tensor([np.random.uniform(-5,5), np.random.uniform(-5,5),
                               np.random.uniform(-np.pi,np.pi), np.random.uniform(0,5),
                               np.random.uniform(-2,2)], device=device)
        F_opt, ts = solveForcingWithGrad(v0=v0, target=target, t0=t0, t1=t1, dt_max=dt_max,
                                         nsteps=nsteps, lr=lr, weight_decay=weight_decay,
                                         break_thresh=break_thresh)
        V_mat, _, _ = compute_path_gradient(v0, F_opt.to(device), ts.to(device), target)
        inp = torch.cat([V_mat[:-1], F_opt.to(device)], dim=1).cpu().numpy()
        out = torch.cat([V_mat[1:], F_opt.to(device)], dim=1).cpu().numpy()
        sequences.append(inp)
        targets.append(out)
    return sequences, targets

# --- Neural Network (RNN) Components ---
class TrajectoryDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        self.targets = [torch.tensor(t, dtype=torch.float32) for t in targets]
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.targets[idx]

class TrajectoryRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_layers=1, rnn_type='GRU', dropout=0.0):
        super().__init__()
        if rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)


def train_rnn(model, dataloader, num_epochs=50, lr=1e-3, device=device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(1, num_epochs+1):
        start_ep = time.time()
        total_loss = 0.0
        for seqs, targets in dataloader:
            seqs, targets = seqs.to(device), targets.to(device)  
            preds = model(seqs)  
            loss = criterion(preds, targets)  
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
            total_loss += loss.item() * seqs.size(0)
        avg = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch:03d} | Loss: {avg:.6f} | Time: {time.time()-start_ep:.2f}s", flush=True)

# --- Main execution ---
if __name__ == '__main__':
    print(f"Working directory: {os.getcwd()}")
    print(f"Running on device: {device}\n")

    # --- Random single-point solver test ---
    v0 = torch.tensor([np.random.uniform(-5,5), np.random.uniform(-5,5),
                        np.random.uniform(-np.pi,np.pi), np.random.uniform(0,5),
                        np.random.uniform(-2,2)], device=device)
    target = torch.tensor([np.random.uniform(-5,5), np.random.uniform(-5,5),
                           np.random.uniform(-np.pi,np.pi), np.random.uniform(0,5),
                           np.random.uniform(-2,2)], device=device)
    print(f"Solver test with v0={v0.cpu().numpy()} target={target.cpu().numpy()}\n")
    solveForcingWithGrad(v0=v0, target=target, nsteps=200, lr=1e-2)

    # --- Load or generate random training data ---
    data_file = 'training_data.npz'
    if os.path.exists(data_file):
        print(f"Loading training data from {data_file}\n")
        data = np.load(data_file)
        seq_arr = data['sequences']
        tar_arr = data['targets']
        sequences = [seq_arr[i] for i in range(seq_arr.shape[0])]
        targets = [tar_arr[i] for i in range(tar_arr.shape[0])]
    else:
        num_samples = 5
        print(f"Generating {num_samples} random training samples...\n")
        sequences, targets = generate_training_data(n_samples=num_samples,
                                                    t0=0, t1=5.5, dt_max=0.25,
                                                    nsteps=100, lr=6e-2,
                                                    weight_decay=1e-1,
                                                    break_thresh=1e-5)
        seq_arr = np.stack(sequences)
        tar_arr = np.stack(targets)
        np.savez(data_file, sequences=seq_arr, targets=tar_arr)
        print(f"Saved training data to {data_file}\n")

    # --- Train RNN on data ---
    dataset = TrajectoryDataset(sequences, targets)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = TrajectoryRNN(input_size=7, hidden_size=64,
                           output_size=7, num_layers=1, rnn_type='GRU')
    print("Training RNN on random data...\n")
    train_rnn(model, loader, num_epochs=10, lr=1e-3)
    print("Done!")
