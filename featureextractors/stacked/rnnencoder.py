import torch
from torch import nn


class RNNForStackedFeatures(nn.Module):
    def __init__(self, sample_input: torch.Tensor, input_embedding_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 output_dim: int = 128, rnn_type: str = 'lstm'):
        super().__init__()

        # Define the RNN based on the type specified
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(input_size=input_embedding_dim, hidden_size=hidden_dim, num_layers=num_layers,
                               batch_first=True)
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(input_size=input_embedding_dim, hidden_size=hidden_dim, num_layers=num_layers,
                              batch_first=True)
        else:
            raise ValueError("Invalid RNN type. Choose 'lstm' or 'gru'.")

        with torch.no_grad():
            _, (h_n, _) = self.rnn(sample_input)
            n_flatten = h_n.shape[2]

        self.linear = nn.Sequential(nn.Linear(n_flatten, output_dim), nn.ReLU())

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # RNN forward pass
        rnn_output, (h_n, c_n) = self.rnn(x)

        # aggregated_output dim is [batch, hidden_dim]
        aggregated_output = h_n[-1]  # Use the last hidden state
        x = self.linear(aggregated_output)
        return x
