import torch
from torch import nn
import gymnasium as gym

from featureextractors.CNNfeatureextractor import CNNFeatureExtractor


class AgentIdFeatureExtractor(nn.Module):
    def __init__(self, observation_space: gym.Space, output_dim: int):
        super().__init__()
        # Adding a channel and batch dim to sample observations
        self.CNNFeatureExtractor = CNNFeatureExtractor(embedding_dim=1, cnn_output_dim=output_dim,
                                                       sample_observations=torch.as_tensor(
                                                           observation_space.sample()[None][..., None]).permute(0, 3, 1,
                                                                                                                2))

    def forward(self, x):
        x = x[..., None]  # adding a channel dim

        if x.dim() == 4:
            # [batch_size, height, width, channels] -> [batch_size, channels, height, width]
            x = x.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unsupported embedding tensor shape: {x.shape}")

        return self.CNNFeatureExtractor(x)
