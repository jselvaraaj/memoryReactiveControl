import gymnasium as gym
import torch
from torch import nn

from featureextractors.CNNfeatureextractor import CNNFeatureExtractor


class AgentIdFeatureExtractor(nn.Module):
    def __init__(self, observation_space: gym.Space, config: dict):
        super().__init__()

        output_dim = config['output_dim']
        # Adding a channel and batch dim to sample observations
        self.CNNFeatureExtractor = CNNFeatureExtractor(cnn_output_dim=output_dim,
                                                       sample_observations=self.get_embeddings(torch.as_tensor(
                                                           observation_space.sample()[None]))
                                                       , config=config)

    def get_embeddings(self, x):
        x = x[..., None]
        if x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unsupported embedding tensor shape: {x.shape}")
        return x

    def forward(self, x):
        return self.CNNFeatureExtractor(self.get_embeddings(x))
