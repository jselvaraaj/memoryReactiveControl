from typing import Dict

import torch
import gymnasium as gym
from common.type_aliases import TensorDict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from torch import nn


class GridVerseFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.vector.utils.spaces.Dict,
            grid_embedding_dim: int = 8,
            cnn_output_dim: int = 256,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == 'grid':
                # The i,j is grid co-ordinates. the last index is the grid object index type; for now we are ignoring the color and items in the grid
                number_of_objects = subspace.high[0, 0, 0] - subspace.low[0, 0, 0] + 1
                extractors[key] = GridFeatureExtractor(number_of_objects, grid_embedding_dim, cnn_output_dim, subspace)
                total_concat_size += cnn_output_dim
            elif key == 'agent_id':
                extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim, normalized_image=False)
                total_concat_size += cnn_output_dim

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)


class GridFeatureExtractor(nn.Module):
    def __init__(self, number_of_objects: int, grid_embedding_dim: int, cnn_output_dim: int,
                 observation_space: gym.Space):
        super().__init__()
        self.embedding = nn.Embedding(number_of_objects, grid_embedding_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(grid_embedding_dim, 32, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
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

    # Assuming input tensor order [batch_size, height, width, channels]
    def forward(self, x):
        x = x[..., 0].to(torch.int)  # just taking the grid object index type ignoring the color and items
        embedding = self.embedding(x).permute(0, 3, 1, 2)
        x = self.linear(self.cnn(embedding))
        return x
