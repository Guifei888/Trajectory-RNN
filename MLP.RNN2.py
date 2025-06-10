import torch as pt
import torch.nn as nn
from torch.optim import AdamW
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import importlib.util, sys, os

# Dynamically load your original MLP.RNN.py module
spec = importlib.util.spec_from_file_location(
    "mlp_orig",
    os.path.join(os.path.dirname(__file__), "MLP.RNN.py")
)
mlp_orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlp_orig)

# Bring in your definitions
MLP                   = mlp_orig.MLP
create_ics            = mlp_orig.create_ics
ICSDataset            = mlp_orig.ICSDataset
compute_path_autograd = mlp_orig.compute_path_autograd
showTraj              = mlp_orig.showTraj

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

class MLPWithRNN(nn.Module):
    """
    End-to-end model that first predicts a one-shot forcing sequence via an MLP,
    then refines it with a single GRU pass predicting deltas (ΔF) for each time step.
    """
    def __init__(self,
                 mlp: nn.Module,
                 rnn_hidden: int = 64,
                 bidirectional: bool = False):
        super().__init__()
        self.mlp = mlp
        # feature dimension per step (phi, psi, dt)
        self.feature_dim = 3

        # Single-layer GRU for refinement
        self.rnn = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )
        out_dim = rnn_hidden * (2 if bidirectional else 1)
        self.delta_proj = nn.Linear(out_dim, self.feature_dim)

    def forward(self, v0: pt.Tensor, target: pt.Tensor) -> pt.Tensor:
        """
        v0:     Tensor of shape (batch, 5)
        target: Tensor of shape (batch, 5)
        returns: F_refined of shape (batch, time_steps, 3)
        """
        # 1) One-shot forcing from your existing MLP
        F_init = self.mlp(v0, target).to(device)  # (batch, T, 3)

        # 2) RNN-based delta prediction
        rnn_out, _ = self.rnn(F_init)            # (batch, T, hidden)
        delta = self.delta_proj(rnn_out)         # (batch, T, 3)

        # 3) Refined forces
        F_refined = F_init + delta
        return F_refined

if __name__ == '__main__':
    # Hyperparameters
    BATCH = 4
    SHAPE = (64,)

    # Prepare data
    ics = create_ics(shape=SHAPE)
    dataset = ICSDataset(ics)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)
    v0_batch, target_batch = next(iter(loader))
    v0_batch, target_batch = v0_batch.to(device), target_batch.to(device)

    # Instantiate models
    mlp = MLP(input_dim=5, output_dim=32, latent_dim=32).to(device)
    model = MLPWithRNN(mlp, rnn_hidden=64, bidirectional=False).to(device)

    # Forward pass
    F_combined = model(v0_batch, target_batch)

    # Simulate
    V_comb, _ = compute_path_autograd(v0_batch, F_combined, target_batch)

    # Plot trajectory
    print("Plotting combined MLP+RNN trajectory...")
    showTraj(V_comb, F_combined)

    # Plot forcing sequence
    F_np = F_combined.detach().cpu().numpy()[0]
    phi, psi, dt = F_np[:,0], F_np[:,1], F_np[:,2]
    plt.figure()
    plt.plot(phi, label='phi')
    plt.plot(psi, label='psi')
    plt.plot(dt,  label='dt')
    plt.title('MLP+RNN Refined Forcing Sequence')
    plt.xlabel('Time Step')
    plt.legend()
    plt.show()
