import gymnasium as gym
import torch
from torch import nn

from gridverse_torch_featureextractors.CNNfeatureextractor import CNNFeatureExtractor


class GridFeatureExtractor(nn.Module):
    def __init__(self, number_of_grid_cell_types: int, number_of_grid_cell_states_per_cell_type: int,
                 number_of_grid_colors: int, config: dict,
                 observation_space: gym.Space):
        super().__init__()

        embedding_dim = config['embedding_dim']
        cnn_output_dim = config['output_dim']
        self.number_of_grid_cell_types = number_of_grid_cell_types
        self.number_of_grid_cell_states_per_cell_type = number_of_grid_cell_states_per_cell_type
        self.number_of_grid_colors = number_of_grid_colors

        self.embedding = nn.Embedding(
            number_of_grid_cell_types * number_of_grid_cell_states_per_cell_type * number_of_grid_colors, embedding_dim)

        self.cnn = CNNFeatureExtractor(cnn_output_dim=cnn_output_dim,
                                       sample_observations=self.get_embeddings(torch.as_tensor(
                                           observation_space.sample()[None])),
                                       config=config)

    def get_embeddings(self, x):
        modified_index = x[..., 0] + \
                         x[..., 1] * self.number_of_grid_cell_types + \
                         x[..., 2] * self.number_of_grid_cell_types * self.number_of_grid_cell_states_per_cell_type
        embedding = self.embedding(modified_index.to(torch.int))

        if embedding.dim() == 4:
            # [batch_size, height, width, embedding_dim/channels] -> [batch_size, embedding_dim/channels, height, width]
            embedding = embedding.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unsupported embedding tensor shape: {embedding.shape}")

        return embedding

    def forward(self, x):
        embedding = self.get_embeddings(x)
        x = self.cnn(embedding)
        return x
