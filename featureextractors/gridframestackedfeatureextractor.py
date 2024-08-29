import torch
from torch import nn
import gymnasium as gym
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from featureextractors.stackedCNNfeatureextractor import StackedCNNFeatureExtractor


class GridFrameStackedFeatureExtractor(nn.Module):
    def __init__(self,
                 observation_space: gym.Space, number_of_objects: int, grid_embedding_dim: int, cnn_output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(number_of_objects, grid_embedding_dim)

        sample_obs = self.get_embedding(torch.as_tensor(observation_space.sample()[None]))
        self.stackedCNNFeatureExtractor = StackedCNNFeatureExtractor(embedding_dim=grid_embedding_dim,
                                                                     cnn_output_dim=cnn_output_dim,
                                                                     sample_observations=sample_obs)

    def get_embedding(self, x):
        x = x[..., 0].to(torch.int)  # just taking the grid object index type ignoring the color and items
        return self.embedding(x)

    def forward(self, x):
        x = self.get_embedding(x)
        return self.stackedCNNFeatureExtractor(x)
