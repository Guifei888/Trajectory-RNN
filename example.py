class ForcingRNN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.rnn = nn.GRU(input_size=7, hidden_size=hidden_size,
                          num_layers=2, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 2)  # predict Δϕ, Δψ

    def forward(self, V, F_prev, T):
        """
        V: (batch, seq_len, 5) states
        F_prev: (batch, seq_len, 2) previous forces
        T: (batch, 5) target, expanded to (batch, seq_len, 5)
        """
        # Concatenate [state, prev_force, target] → (batch, seq_len, 12)
        T_exp = T.unsqueeze(1).expand(-1, V.size(1), -1)
        x = torch.cat([V, F_prev, T_exp], dim=-1)
        
        h, _ = self.rnn(x)                 # (batch, seq_len, hidden)
        delta_f = self.fc(h)               # (batch, seq_len, 2)
        return F_prev + delta_f            # refined forcing in one shot