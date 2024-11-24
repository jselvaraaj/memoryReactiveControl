import gymnasium as gym
import torch
from torch import nn

from gridverse_torch_featureextractors.stacked.CNNfeatureextractor import StackedCNNFeatureExtractor


class GridFrameStackedFeatureExtractor(nn.Module):
    def __init__(self,
                 observation_space: gym.Space, number_of_objects: int, grid_embedding_dim: int, cnn_output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(number_of_objects, grid_embedding_dim)

        # [batch, seq_len,x,y,embedding_dim] . Note we are going to treat embdeding_dim as channels inside the CNN
        # So this is equivalent to [batch, seq_len, x, y, channels]
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
