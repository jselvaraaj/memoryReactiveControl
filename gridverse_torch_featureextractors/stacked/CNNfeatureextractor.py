import torch
from torch import nn, Tensor
import gymnasium as gym


class StackedCNNFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim: int, cnn_output_dim: int, sample_observations: Tensor):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(embedding_dim, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            # nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            # nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_observations = sample_observations.to(torch.float32)
            batch_size, seq_len, channels, height, width = sample_observations.size()
            n_flatten = \
                self.cnn(sample_observations.view(batch_size * seq_len, channels, height, width)
                         .permute(0, 3, 1, 2)).view(batch_size, seq_len, -1).shape[2]

        self.linear = nn.Sequential(nn.Linear(n_flatten, cnn_output_dim), nn.ReLU())

    def forward(self, x):
        x = x.to(torch.float32)
        if x.dim() == 5:
            # [batch_size, seq_len, height, width, channels] -> [batch_size, seq_len, channels, height, width]
            embedding = x.permute(0, 1, 4, 2, 3)
        else:
            raise ValueError(f"Unsupported embedding tensor shape: {x.shape}")

        batch_size, seq_len, channels, height, width = embedding.size()

        # Reshape obs to process each image in the sequence through the CNN
        embedding = embedding.view(batch_size * seq_len, channels, height, width)
        cnn_features = self.cnn(embedding)

        # Reshape back to sequence format [batch_size, seq_len, feature_dim]
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        return self.linear(cnn_features)
