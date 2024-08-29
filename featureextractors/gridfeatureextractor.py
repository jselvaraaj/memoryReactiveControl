import torch
from torch import nn
import gymnasium as gym

class GridFeatureExtractor(nn.Module):
    def __init__(self, number_of_objects: int, grid_embedding_dim: int, cnn_output_dim: int,
                 observation_space: gym.Space):
        super().__init__()
        self.embedding = nn.Embedding(number_of_objects, grid_embedding_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(grid_embedding_dim, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            # nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            # nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = \
                self.cnn(
                    self.embedding(torch.as_tensor(observation_space.sample()[..., 0][None])).permute(0, 3, 1,
                                                                                                      2)).shape[
                    1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, cnn_output_dim), nn.ReLU())

    def forward(self, x):
        x = x[..., 0].to(torch.int)  # just taking the grid object index type ignoring the color and items
        embedding = self.embedding(x)

        if embedding.dim() == 4:
            # [batch_size, height, width, channels] -> [batch_size, channels, height, width]
            embedding = embedding.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unsupported embedding tensor shape: {embedding.shape}")

        x = self.linear(self.cnn(embedding))
        return x
