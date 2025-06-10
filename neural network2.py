import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import tqdm  # Import tqdm for progress tracking

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- Base Dynamics & Simulation Helpers ---

def v_prime_components(v, dt):
    """
    Compute Taylor-series components:
        v' = A(v, dt) + phi * B_phi(v, dt) + psi * B_psi(v, dt)
    Returns A, B_phi, B_psi tensors matching the shape of v.
    Optimized to avoid unnecessary operations.
    """
    x, y, vx, vy, theta = torch.unbind(v, dim=-1)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)

    # Drift update (A)
    A = torch.stack([x + vx * dt,
                     y + vy * dt,
                     vx,
                     vy,
                     theta], dim=-1)

    # Control basis for forward acceleration (B_phi)
    B_phi = torch.zeros_like(v)
    B_phi[..., 2] = cos_t * dt
    B_phi[..., 3] = sin_t * dt

    # Control basis for angular acceleration (B_psi)
    B_psi = torch.zeros_like(v)
    B_psi[..., 4] = dt

    return A, B_phi, B_psi

@torch.no_grad()  # Add no_grad decorator for sim functions used during data generation
def simulate(v0, F_seq, ts):
    """Optimized discrete integrator using v_prime_components."""
    batch = v0.dim() > 1
    if batch:
        B = v0.shape[0]
        T = len(ts)
        states = [v0]
        current_v = v0
        for i in range(T - 1):
            dt = ts[i+1] - ts[i]
            A, B_phi, B_psi = v_prime_components(current_v, dt)
            phi = F_seq[:, i, 0]
            psi = F_seq[:, i, 1]
            next_v = A + phi.unsqueeze(-1) * B_phi + psi.unsqueeze(-1) * B_psi
            states.append(next_v)
            current_v = next_v
        V = torch.stack(states, dim=1)
    else:
        T = len(ts)
        states = [v0]
        current_v = v0
        for i in range(T - 1):
            dt = ts[i+1] - ts[i]
            A, B_phi, B_psi = v_prime_components(current_v, dt)
            phi, psi = F_seq[i]
            next_v = A + phi * B_phi + psi * B_psi
            states.append(next_v)
            current_v = next_v
        V = torch.stack(states, dim=0)
    return V

def consistent_simulation(v0, F_seq, ts):
    """Alias to simulate for clarity."""
    return simulate(v0, F_seq, ts)

@torch.no_grad()  # Add no_grad decorator for validation function
def create_valid_trajectory(v0, F_gt, ts, xT, yT, tol=3.0):
    """
    Simulate and check if final (x,y) is within tol of target.
    Returns a bool mask, the full trajectory tensor, and endpoint error.
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
        error = torch.norm(endpoint - target)
        valid = error < tol
    return valid, sim_traj, error.item()

# --- Visualization ---

def visualize_trajectories(examples, xT, yT, tol, num_to_plot=5, save_path="trajectories.png"):
    """
    Plot (x, y) trajectories, start, target, and tolerance region.
    """
    plt.figure(figsize=(8, 8))
    
    num_to_plot = min(num_to_plot, len(examples))
    for i in range(num_to_plot):
        v0, F_gt, ts, _ = examples[i]
        traj = consistent_simulation(v0, F_gt, ts)
        traj_np = traj.detach().cpu().numpy()
        plt.plot(traj_np[:, 0], traj_np[:, 1], 'b-', alpha=0.5, label='Trajectories' if i == 0 else None)
        plt.plot(traj_np[-1, 0], traj_np[-1, 1], 'bo', label='Endpoints' if i == 0 else None)
    
    plt.plot(0, 0, 'go', label='Start (0, 0)')
    plt.plot(xT, yT, 'r*', label=f'Target ({xT:.2f}, {yT:.2f})')
    circle = plt.Circle((xT, yT), tol, color='r', fill=False, linestyle='--', label='Tolerance')
    plt.gca().add_patch(circle)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Simulated Ground-Truth Trajectories')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(save_path)
    plt.close()  # Close the plot to avoid memory leak
    print(f"Ground-truth trajectory plot saved to {save_path}")

def visualize_rnn_vs_gt(model, example, xT, yT, tol, device, save_path="rnn_vs_gt_trajectories.png", title="Ground-Truth vs. RNN-Predicted Trajectory"):
    """
    Plot ground-truth vs. RNN-predicted trajectory for a single example.
    """
    model.eval()
    with torch.no_grad():
        v0, F_gt, ts, _ = example
        v0 = v0.to(device)
        F_gt = F_gt.to(device)
        ts = ts.to(device)

        gt_traj = consistent_simulation(v0, F_gt, ts)
        gt_traj_np = gt_traj.detach().cpu().numpy()

        T = F_gt.shape[0]
        D = v0.shape[-1]
        dummy_states = torch.zeros((1, T, D), device=device)
        dummy_controls = torch.zeros_like(F_gt).unsqueeze(0)
        start_time = time.perf_counter()
        F_init, _ = model(dummy_states, dummy_controls)
        end_time = time.perf_counter()
        pred_time = end_time - start_time
        F_init = F_init.squeeze(0)
        rnn_traj = consistent_simulation(v0, F_init, ts)
        rnn_traj_np = rnn_traj.detach().cpu().numpy()

        deltaF = F_gt - F_init
        mae = deltaF.abs().mean().item()
        endpoint_error = np.linalg.norm(gt_traj_np[-1, :2] - rnn_traj_np[-1, :2])

        plt.figure(figsize=(8, 8))
        plt.plot(gt_traj_np[:, 0], gt_traj_np[:, 1], 'b-', label='Ground-Truth Trajectory')
        plt.plot(gt_traj_np[-1, 0], gt_traj_np[-1, 1], 'bo', label='Ground-Truth Endpoint')
        plt.plot(rnn_traj_np[:, 0], rnn_traj_np[:, 1], 'r--', label='RNN-Predicted Trajectory')
        plt.plot(rnn_traj_np[-1, 0], rnn_traj_np[-1, 1], 'ro', label='RNN-Predicted Endpoint')
        plt.plot(v0[0].item(), v0[1].item(), 'go', label=f'Start ({v0[0].item()}, {v0[1].item()})')
        plt.plot(xT, yT, 'r*', label=f'Target ({xT:.2f}, {yT:.2f})')
        circle = plt.Circle((xT, yT), tol, color='r', fill=False, linestyle='--', label='Tolerance')
        plt.gca().add_patch(circle)
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'{title}\nMAE: {mae:.4f}, Endpoint Error: {endpoint_error:.2f}, Pred Time: {pred_time*1000:.2f} ms')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(save_path)
        plt.close()  # Close the plot to avoid memory leak
        print(f"Plot saved to {save_path}")
        print(f"MAE: {mae:.4f}, Endpoint Error: {endpoint_error:.2f}, Prediction Time: {pred_time*1000:.2f} ms")
        return mae, endpoint_error, pred_time

# --- Model & Dataset ---

class TrajectoryRNN(nn.Module):
    def __init__(self, state_dim=5, control_dim=2, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        # Reduced hidden_dim and num_layers for faster execution
        self.rnn = nn.GRU(state_dim + control_dim,
                          hidden_dim,
                          num_layers,
                          batch_first=True,
                          dropout=dropout)
        self.fc = nn.Linear(hidden_dim, control_dim)
        
        # Initialize weights to help convergence
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

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
    
    # Add progress bar
    pbar = tqdm.tqdm(loader, desc="Training")

    for v0, F_gt, ts, (xT, yT) in pbar:
        v0 = v0.to(device)
        F_gt = F_gt.to(device)
        ts = ts.to(device)

        B = v0.shape[0] if v0.dim() > 1 else 1
        T = F_gt.shape[1] if F_gt.dim() > 1 else F_gt.shape[0]
        D = v0.shape[-1]

        dummy_states = torch.zeros((B, T, D), device=device)
        dummy_controls = torch.zeros_like(F_gt)

        optimizer.zero_grad()
        start_time = time.perf_counter()
        F_init, _ = model(dummy_states, dummy_controls)
        end_time = time.perf_counter()
        pred_time = end_time - start_time

        deltaF = F_gt - F_init
        loss = deltaF.abs().mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mae += deltaF.abs().mean().item()
        total_time += pred_time
        num_batches += 1
        
        # Update progress bar with current loss
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    pbar.close()
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    avg_time = total_time / num_batches
    return avg_loss, avg_mae, avg_time

# --- Helper function for trajectory generation ---
def generate_trajectory(v0, ts, xT, yT, max_iter=30, lr=5e-3, early_stop_threshold=0.05):
    """Optimized trajectory generation with early stopping"""
    F_gt = torch.zeros((len(ts), 2), requires_grad=True)
    optimizer_F = optim.Adam([F_gt], lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_F, patience=5, factor=0.5, verbose=False)
    
    best_error = float('inf')
    best_F = None
    
    for i in range(max_iter):
        optimizer_F.zero_grad()
        traj = consistent_simulation(v0, F_gt, ts)
        error = torch.norm(traj[-1, :2] - torch.tensor([xT, yT]))
        
        # Save best solution
        if error.item() < best_error:
            best_error = error.item()
            best_F = F_gt.detach().clone()
            
            # Early stopping if error is below threshold
            if best_error < early_stop_threshold:
                break
                
        error.backward()
        optimizer_F.step()
        scheduler.step(error)
    
    # Use best found solution
    return best_F if best_F is not None else F_gt.detach()

# --- Main Script ---

if __name__ == '__main__':
    # Use CPU if available, otherwise be explicit about which GPU to use
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # Specify GPU index if needed
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Smaller, faster model
    model = TrajectoryRNN(hidden_dim=128, num_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    model_path = "trajectory_rnn.pth"
    log_file = "training_log.txt"
    run_number = 1

    # Find the latest model file
    while os.path.exists(f"trajectory_rnn_run{run_number}.pth"):
        run_number += 1
    if run_number > 1:
        latest_model_path = f"trajectory_rnn_run{run_number-1}.pth"
        try:
            model.load_state_dict(torch.load(latest_model_path, map_location=device))
            print(f"Loaded existing model from {latest_model_path}")
        except Exception as e:
            print(f"Error loading model: {e}. Starting with a fresh model.")
    else:
        print("No existing model found. Starting with a fresh model.")

    # Generate training dataset (reduced size for faster execution)
    examples = []
    num_examples = 500  # Reduced from 2000
    ts = torch.linspace(0, 10, steps=25)  # Reduced from 50
    T = len(ts)
    xT, yT = 5.68071168, 2.5029068
    tol = 3.0
    debug_log = "dataset_debug.txt"

    with open(debug_log, 'w') as f:
        f.write("Debugging trajectory generation:\n")

    print(f"Generating {num_examples} trajectories...")
    
    # Progress bar for trajectory generation
    for i in tqdm.tqdm(range(num_examples), desc="Generating trajectories"):
        v0 = torch.zeros(5)
        v0[2] = 0.2 + torch.randn(1) * 0.1  # Initial vx
        v0[3] = 1.0 + torch.randn(1) * 0.1  # Initial vy
        theta = torch.atan2(torch.tensor(yT), torch.tensor(xT)) + torch.randn(1) * 0.1
        v0[4] = theta

        # Use optimized trajectory generation
        F_gt = generate_trajectory(v0, ts, xT, yT)

        valid, traj, error = create_valid_trajectory(v0, F_gt, ts, xT, yT, tol=tol)
        with open(debug_log, 'a') as f:
            f.write(f"Trajectory {i+1}: Valid={valid}, Endpoint Error={error:.2f}\n")
        if valid:
            examples.append((v0, F_gt, ts, (xT, yT)))
        
        # Break early if we have enough examples
        if len(examples) >= num_examples // 2:
            break

    if len(examples) == 0:
        raise ValueError(f"No valid trajectories generated. Check control inputs or tolerance. See {debug_log} for details.")

    print(f"Generated {len(examples)} valid trajectories.")
    
    try:
        visualize_trajectories(examples[:5], xT, yT, tol, num_to_plot=5, save_path=f"trajectories_run{run_number}.png")
    except Exception as e:
        print(f"Warning: Unable to generate visualization: {e}")

    # Train model with progress output
    loader = DataLoader(TrajectoryDataset(examples), batch_size=16, shuffle=True)
    print("\nTraining model...")
    num_epochs = 30  # Reduced from 100
    
    for epoch in range(num_epochs):
        avg_loss, avg_mae, avg_time = train_epoch(model, loader, optimizer, device)
        scheduler.step()
        print(f"Run {run_number} | Epoch {epoch:03d}/{num_epochs} | Loss: {avg_loss:.4f} | MAE: {avg_mae:.4f} | Avg Prediction Time: {avg_time*1000:.2f} ms")
        
        # Save intermediate model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            intermediate_path = f"trajectory_rnn_run{run_number}_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), intermediate_path)
            print(f"Saved intermediate model to {intermediate_path}")

    new_model_path = f"trajectory_rnn_run{run_number}.pth"
    torch.save(model.state_dict(), new_model_path)
    print(f"Model saved to {new_model_path}")

    # Test model with a single example
    print("\nTesting model performance...")
    test_v0 = torch.tensor([0., 0., 0.2, 1., 1.], dtype=torch.float32)
    test_xT = 5.68071168
    test_yT = 2.5029068
    test_tol = 3.0

    test_theta = torch.atan2(torch.tensor(test_yT), torch.tensor(test_xT))
    test_v0 = torch.tensor([0., 0., 0.2, 1., test_theta.item()], dtype=torch.float32)
    
    # Generate optimized test trajectory
    test_F_gt = generate_trajectory(test_v0, ts, test_xT, test_yT, max_iter=50)

    valid, test_traj, error = create_valid_trajectory(test_v0, test_F_gt, ts, test_xT, test_yT, tol=test_tol)
    if not valid:
        print(f"Warning: Test trajectory not valid. Endpoint Error: {error:.2f}")
    test_example = (test_v0, test_F_gt, ts, (test_xT, test_yT))

    try:
        mae, endpoint_error, pred_time = visualize_rnn_vs_gt(
            model,
            test_example,
            test_xT,
            test_yT,
            test_tol,
            device,
            save_path=f"test_rnn_vs_gt_run{run_number}.png",
            title=f"Ground-Truth vs. RNN-Predicted Trajectory (Test ICs, Run {run_number})"
        )

        with open(log_file, 'a') as f:
            f.write(f"Run {run_number} | MAE: {mae:.4f} | Endpoint Error: {endpoint_error:.2f} | Prediction Time: {pred_time*1000:.2f} ms\n")
    except Exception as e:
        print(f"Warning: Unable to generate test visualization: {e}")
        
    print("Done!")