import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# --- Base Dynamics & Simulation Helpers ---

def v_prime_components(v, dt):
    """
    Compute Taylor-series components:
        v' = A(v, dt) + phi * B_phi(v, dt) + psi * B_psi(v, dt)
    Returns A, B_phi, B_psi tensors matching the shape of v.
    """
    x, y, vx, vy, theta = torch.unbind(v, dim=-1)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)

    # Drift update (A)
    A = torch.stack([x + vx * dt,
                     y + vy * dt,
                     vx,
                     vy,
                     theta], dim=-1)

    # Control basis for forward acceleration
    B_phi = torch.zeros_like(v)
    B_phi[..., 2] = cos_t * dt
    B_phi[..., 3] = sin_t * dt

    # Control basis for angular acceleration
    B_psi = torch.zeros_like(v)
    B_psi[..., 4] = dt

    return A, B_phi, B_psi

def simulate(v0, F_seq, ts):
    """Simple discrete integrator using v_prime_components."""
    batch = v0.dim() > 1
    if batch:
        B = v0.shape[0]
        T = len(ts)
        V = torch.zeros((B, T, 5), device=v0.device, dtype=v0.dtype)
        V[:, 0] = v0
    else:
        T = len(ts)
        V = torch.zeros((T, 5), device=v0.device, dtype=v0.dtype)
        V[0] = v0

    for i in range(T - 1):
        dt = ts[i+1] - ts[i]
        if batch:
            A, B_phi, B_psi = v_prime_components(V[:, i], dt)
            phi = F_seq[:, i, 0]; psi = F_seq[:, i, 1]
            V[:, i+1] = A + phi.unsqueeze(-1) * B_phi + psi.unsqueeze(-1) * B_psi
        else:
            A, B_phi, B_psi = v_prime_components(V[i], dt)
            phi, psi = F_seq[i]
            V[i+1] = A + phi * B_phi + psi * B_psi
    return V

def consistent_simulation(v0, F_seq, ts):
    """Alias to simulate for clarity—use this everywhere."""
    return simulate(v0, F_seq, ts)

def create_valid_trajectory(v0, F_gt, ts, xT, yT, tol=2.0):
    """
    Simulate and check if final (x,y) is within tol of target.
    Returns a bool mask and the full trajectory tensor.
    """
    sim_traj = consistent_simulation(v0, F_gt, ts)
    if sim_traj.dim() == 3:
        endpoint = sim_traj[:, -1, :2]
        target = torch.tensor([xT, yT], device=v0.device).unsqueeze(0)
        error = torch.norm(endpoint - target, dim=1)
        valid = error < tol
    else:
        endpoint = sim_traj[-1, :2]
        target = torch.tensor([xT, yT], device=v0.device)
        valid = torch.norm(endpoint - target) < tol
    return valid, sim_traj

# --- Visualization ---

def visualize_trajectories(examples, xT, yT, tol, num_to_plot=5, save_path="trajectories.png"):
    """
    Plot (x, y) trajectories, start, target, and tolerance region.
    """
    plt.figure(figsize=(8, 8))
    
    # Plot a subset of trajectories
    num_to_plot = min(num_to_plot, len(examples))
    for i in range(num_to_plot):
        v0, F_gt, ts, _ = examples[i]
        traj = consistent_simulation(v0, F_gt, ts)
        traj_np = traj.detach().cpu().numpy()
        plt.plot(traj_np[:, 0], traj_np[:, 1], 'b-', alpha=0.5, label='Trajectories' if i == 0 else None)
        plt.plot(traj_np[-1, 0], traj_np[-1, 1], 'bo', label='Endpoints' if i == 0 else None)
    
    # Plot start and target
    plt.plot(0, 0, 'go', label='Start (0, 0)')
    plt.plot(xT, yT, 'r*', label=f'Target ({xT}, {yT})')
    
    # Plot tolerance circle
    circle = plt.Circle((xT, yT), tol, color='r', fill=False, linestyle='--', label='Tolerance')
    plt.gca().add_patch(circle)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Simulated Ground-Truth Trajectories')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(save_path)
    plt.show()
    print(f"Ground-truth trajectory plot saved to {save_path}")

def visualize_rnn_vs_gt(model, example, xT, yT, tol, device, save_path="rnn_vs_gt_trajectories.png"):
    """
    Plot ground-truth vs. RNN-predicted trajectory for a single example.
    """
    model.eval()
    with torch.no_grad():
        v0, F_gt, ts, _ = example
        v0 = v0.to(device)
        F_gt = F_gt.to(device)
        ts = ts.to(device)

        # Ground-truth trajectory
        gt_traj = consistent_simulation(v0, F_gt, ts)
        gt_traj_np = gt_traj.detach().cpu().numpy()

        # RNN-predicted trajectory
        T = F_gt.shape[0]
        D = v0.shape[-1]
        dummy_states = torch.zeros((1, T, D), device=device)
        dummy_controls = torch.zeros_like(F_gt).unsqueeze(0)
        F_init, _ = model(dummy_states, dummy_controls)
        F_init = F_init.squeeze(0)  # Remove batch dimension
        rnn_traj = consistent_simulation(v0, F_init, ts)
        rnn_traj_np = rnn_traj.detach().cpu().numpy()

        # Plot
        plt.figure(figsize=(8, 8))
        plt.plot(gt_traj_np[:, 0], gt_traj_np[:, 1], 'b-', label='Ground-Truth Trajectory')
        plt.plot(gt_traj_np[-1, 0], gt_traj_np[-1, 1], 'bo', label='Ground-Truth Endpoint')
        plt.plot(rnn_traj_np[:, 0], rnn_traj_np[:, 1], 'r--', label='RNN-Predicted Trajectory')
        plt.plot(rnn_traj_np[-1, 0], rnn_traj_np[-1, 1], 'ro', label='RNN-Predicted Endpoint')
        plt.plot(0, 0, 'go', label='Start (0, 0)')
        plt.plot(xT, yT, 'r*', label=f'Target ({xT}, {yT})')
        circle = plt.Circle((xT, yT), tol, color='r', fill=False, linestyle='--', label='Tolerance')
        plt.gca().add_patch(circle)
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Ground-Truth vs. RNN-Predicted Trajectory')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(save_path)
        plt.show()
        print(f"RNN vs. Ground-Truth plot saved to {save_path}")

# --- Model & Dataset ---

class TrajectoryRNN(nn.Module):
    def __init__(self, state_dim=5, control_dim=2, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.rnn = nn.GRU(state_dim + control_dim,
                          hidden_dim,
                          num_layers,
                          batch_first=True,
                          dropout=dropout)
        self.fc = nn.Linear(hidden_dim, control_dim)

    def forward(self, states, controls, hidden=None):
        x = torch.cat([states, controls], dim=-1)
        out, hidden = self.rnn(x, hidden)
        return self.fc(out), hidden

class TrajectoryDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        v0, F_gt, ts, target = self.examples[idx]
        return v0, F_gt, ts, target

# --- Training Loop ---

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_time = 0.0
    num_batches = 0

    for v0, F_gt, ts, (xT, yT) in loader:
        # Move data
        v0 = v0.to(device)
        F_gt = F_gt.to(device)
        ts = ts.to(device)

        # Extract shapes
        B = v0.shape[0] if v0.dim() > 1 else 1
        T = F_gt.shape[1] if F_gt.dim() > 1 else F_gt.shape[0]
        D = v0.shape[-1]

        # Warm-start zeros
        dummy_states = torch.zeros((B, T, D), device=device)
        dummy_controls = torch.zeros_like(F_gt)

        optimizer.zero_grad()
        # Measure prediction time
        start_time = time.perf_counter()
        F_init, _ = model(dummy_states, dummy_controls)
        end_time = time.perf_counter()
        pred_time = end_time - start_time

        deltaF = F_gt - F_init
        loss = deltaF.abs().mean()  # MAE as loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mae += deltaF.abs().mean().item()  # Redundant with loss, kept for clarity
        total_time += pred_time
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    avg_time = total_time / num_batches
    return avg_loss, avg_mae, avg_time

# --- Main Script ---

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TrajectoryRNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model_path = "trajectory_rnn.pth"

    # Load existing model if available
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded existing model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}. Starting with a fresh model.")
    else:
        print("No existing model found. Starting with a fresh model.")

    # Generate and filter dummy examples
    examples = []
    num_examples = 1000
    ts = torch.linspace(0, 10, steps=50)
    T = len(ts)
    xT, yT = 5.0, 5.0
    tol = 2.0

    for _ in range(num_examples):
        # Initial state: start at origin with zero velocity and orientation
        v0 = torch.zeros(5)  # [x=0, y=0, vx=0, vy=0, theta=0]

        # Generate control inputs to approximately reach (xT, yT)
        distance = torch.sqrt(torch.tensor(xT**2 + yT**2))
        total_time = ts[-1]
        phi = (2 * distance / total_time**2)  # Approximate constant acceleration
        psi = 0.0  # No rotation needed for straight-line motion
        F_gt = torch.zeros((T, 2))
        F_gt[:, 0] = phi  # Forward acceleration
        F_gt[:, 1] = psi  # Angular acceleration

        # Adjust theta in v0 to point toward target
        theta = torch.atan2(torch.tensor(yT), torch.tensor(xT))
        v0[4] = theta  # Set initial orientation

        # Simulate and check validity
        valid, traj = create_valid_trajectory(v0, F_gt, ts, xT, yT, tol=tol)
        if valid:
            examples.append((v0, F_gt, ts, (xT, yT)))

    if len(examples) == 0:
        raise ValueError("No valid trajectories generated. Check control inputs or tolerance.")

    print(f"Generated {len(examples)} valid trajectories.")
    
    # Visualize ground-truth trajectories before training
    visualize_trajectories(examples, xT, yT, tol, num_to_plot=5, save_path="trajectories.png")

    loader = DataLoader(TrajectoryDataset(examples), batch_size=32, shuffle=True)
    for epoch in range(10):
        avg_loss, avg_mae, avg_time = train_epoch(model, loader, optimizer, device)
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | MAE: {avg_mae:.4f} | Avg Prediction Time: {avg_time*1000:.2f} ms")

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Visualize RNN vs. ground-truth trajectory for the first example
    if examples:
        visualize_rnn_vs_gt(model, examples[0], xT, yT, tol, device, save_path="rnn_vs_gt_trajectories.png")