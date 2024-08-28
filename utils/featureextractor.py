from typing import Dict

import torch
import gymnasium as gym
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from torch import nn


class GridVerseFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.vector.utils.spaces.Dict,
            grid_embedding_dim: int = 8,
            output_dim: int = 256,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == 'grid':
                if not isinstance(subspace, gym.spaces.Box):
                    raise ValueError(f"Expected observation space to be of type 'Box', but got: {type(subspace)}")
                # The i,j is grid co-ordinates. the last index is the grid object index type; for now we are ignoring the color and items in the grid
                number_of_objects = subspace.high.flatten()[0] - subspace.low.flatten()[0] + 1
                ndim = subspace.high.ndim
                # ndim = 3 means(x, y, [object,color,item]])
                # ndim = 4 means(num_stacks, x, y, [object,color,item]])
                if ndim == 3:
                    extractors[key] = GridFeatureExtractor(number_of_objects, grid_embedding_dim, output_dim, subspace)
                elif ndim == 4:
                    extractors[key] = GridFrameStackedFeatureExtractor(number_of_objects, grid_embedding_dim, output_dim, subspace)
                else:
                    raise ValueError(f"Unsupported grid tensor shape: {subspace.shape}")
                total_concat_size += output_dim
            elif key == 'agent_id':
                extractors[key] = NatureCNN(subspace, features_dim=output_dim, normalized_image=False)
                total_concat_size += output_dim

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

class GridFrameStackedFeatureExtractor(nn.Module):
    def __init__(self, number_of_objects: int, grid_embedding_dim: int, cnn_output_dim: int,
                 observation_space: gym.Space):
        super().__init__()
        self.embedding = nn.Embedding(number_of_objects, grid_embedding_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(grid_embedding_dim, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = \
                self.cnn(
                    self.embedding(torch.as_tensor(observation_space.sample()[..., 0][None])).permute(0, 1,4,2,3)).shape[
                    1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, cnn_output_dim), nn.ReLU())

    def forward(self, x):
        x = x[..., 0].to(torch.int)  # just taking the grid object index type ignoring the color and items
        embedding = self.embedding(x)

        if embedding.dim() == 5:
            # [batch_size, stacked_k_grids, height, width, channels] -> [batch_size, stacked_k_grids, channels, height, width]
            embedding = embedding.permute(0, 1, 4, 2, 3)
        else:
            raise ValueError(f"Unsupported embedding tensor shape: {embedding.shape}")

        x = self.linear(self.cnn(embedding))
        return x
