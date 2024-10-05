import torch
from torch import nn
import gymnasium as gym


class ItemsFeatureExtractor(nn.Module):
    def __init__(self, number_of_item_types: int, number_of_item_states: int,
                 number_of_item_colors: int,
                 config: dict,
                 observation_space: gym.Space):
        super().__init__()

        embedding_dim = config['embedding_dim']
        layers_dims = config['layers']

        self.number_of_item_types = number_of_item_types
        self.number_of_item_states = number_of_item_states
        self.number_of_item_colors = number_of_item_colors

        self.embedding = nn.Embedding(number_of_item_types * number_of_item_states * number_of_item_colors,
                                      embedding_dim)

        layers = []
        prev_layer_dim = embedding_dim
        for layer_dim in layers_dims:
            layers.append(nn.Linear(prev_layer_dim, layer_dim))
            layers.append(nn.ReLU())
            prev_layer_dim = layer_dim

        self.linear = nn.Sequential(*layers)

    def get_embeddings(self, x):
        modified_index = x[..., 0] + \
                         x[..., 1] * self.number_of_item_types + \
                         x[..., 2] * self.number_of_item_types * self.number_of_item_states
        embedding = self.embedding(modified_index.to(torch.int))

        if embedding.dim() != 2:
            raise ValueError(f"Unsupported embedding tensor shape: {embedding.shape}")

        return embedding

    def forward(self, x):
        embedding = self.get_embeddings(x)
        return self.linear(embedding)
