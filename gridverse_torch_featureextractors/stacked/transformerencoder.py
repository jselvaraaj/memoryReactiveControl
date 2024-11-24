import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class TransformerForStackedFeatures(nn.Module):
    def __init__(self, sample_input: torch.Tensor, input_embedding_dim: int, nhead: int = 8, num_layers: int = 6,
                 output_dim: int = 128,
                 ):
        super().__init__()

        # Define the transformer encoder
        encoder_layer = TransformerEncoderLayer(d_model=input_embedding_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        with torch.no_grad():
            n_flatten = self.transformer_encoder(sample_input).shape[2]

        self.linear = nn.Sequential(nn.Linear(n_flatten, output_dim), nn.ReLU())

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        mask = generate_square_subsequent_mask(seq_len).to(x.device)
        transformer_output = self.transformer_encoder(x, mask=mask)

        aggregated_output = transformer_output[:, -1, :]  # Take the last timestep
        x = self.linear(aggregated_output)
        return x


def generate_square_subsequent_mask(seq_len: int):
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == torch.tensor(1)).transpose(0, 1)
    mask = mask.float().masked_fill(mask == torch.tensor(0), float('-inf')).masked_fill(mask == torch.tensor(1),
                                                                                        float(0.0))
    return mask
