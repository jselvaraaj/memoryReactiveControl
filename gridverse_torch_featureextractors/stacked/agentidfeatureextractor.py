import gymnasium as gym
import torch
from torch import nn

from gridverse_torch_featureextractors.stacked.CNNfeatureextractor import StackedCNNFeatureExtractor


class AgentIdStackedFeatureExtractor(nn.Module):
    def __init__(self, observation_space: gym.Space, output_dim: int):
        super().__init__()

        # Adding a channel and batch dim to sample observations
        self.stackedCNNFeatureExtractor = StackedCNNFeatureExtractor(embedding_dim=1, cnn_output_dim=output_dim,
                                                                     sample_observations=torch.as_tensor(
                                                                         observation_space.sample()[None][..., None]))

    def forward(self, x):
        x = x[..., None]  # adding a channel dim
        return self.stackedCNNFeatureExtractor(x)
