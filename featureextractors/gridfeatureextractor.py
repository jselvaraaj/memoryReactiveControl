import torch
from torch import nn
import gymnasium as gym

from featureextractors.CNNfeatureextractor import CNNFeatureExtractor


class GridFeatureExtractor(nn.Module):
    def __init__(self, number_of_objects: int, grid_embedding_dim: int, cnn_output_dim: int,
                 observation_space: gym.Space, cnn_config):
        super().__init__()
        self.embedding = nn.Embedding(number_of_objects, grid_embedding_dim)

        self.cnn = CNNFeatureExtractor(embedding_dim=grid_embedding_dim, cnn_output_dim=cnn_output_dim,
                                       sample_observations=self.embedding(torch.as_tensor(
                                           observation_space.sample()[..., 0][None])).permute(0, 3, 1,
                                                                                              2),
                                       config=cnn_config)

    def forward(self, x):
        x = x[..., 0].to(torch.int)  # just taking the grid object index type ignoring the color and items
        embedding = self.embedding(x)

        if embedding.dim() == 4:
            # [batch_size, height, width, channels] -> [batch_size, channels, height, width]
            embedding = embedding.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unsupported embedding tensor shape: {embedding.shape}")

        x = self.cnn(embedding)
        return x
