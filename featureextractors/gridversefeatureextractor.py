from typing import Dict
import torch
import gymnasium as gym
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from torch import nn

from featureextractors.agentidfeatureextractor import AgentIdStackedFeatureExtractor
from featureextractors.gridfeatureextractor import GridFeatureExtractor
from featureextractors.gridframestackedfeatureextractor import GridFrameStackedFeatureExtractor
from featureextractors.transformerencoder import TransformerForStackedObservation


class GridVerseFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.vector.utils.spaces.Dict,
            grid_embedding_dim: int = 8,
            cnn_output_dim: int = 256,
            transformer_output_dim: int = 128,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        # Grid
        grid_subspace = observation_space.spaces['grid']
        if not isinstance(grid_subspace, gym.spaces.Box):
            raise ValueError(f"Expected observation space to be of type 'Box', but got: {type(grid_subspace)}")
        # The i,j is grid co-ordinates. the last index is the grid object index type; for now we are ignoring the color and items in the grid
        number_of_objects = grid_subspace.high.flatten()[0] - grid_subspace.low.flatten()[0] + 1
        ndim = grid_subspace.high.ndim
        # ndim = 3 means(x, y, [object,color,item]])
        # ndim = 4 means(num_stacks, x, y, [object,color,item]])
        if ndim == 3:
            self.grid_feature_extractor = GridFeatureExtractor(number_of_objects, grid_embedding_dim, cnn_output_dim,
                                                               grid_subspace)
        elif ndim == 4:
            self.grid_feature_extractor = GridFrameStackedFeatureExtractor(grid_subspace, number_of_objects,
                                                                           grid_embedding_dim,
                                                                           cnn_output_dim)
        else:
            raise ValueError(f"Unsupported grid tensor shape: {grid_subspace.shape}")

        # Agent ID
        agent_id_subspace = observation_space.spaces['agent_id_grid']
        self.agent_id_feature_extractor = AgentIdStackedFeatureExtractor(agent_id_subspace, cnn_output_dim)

        # Transformer
        with torch.no_grad():
            sample_grid = torch.as_tensor(grid_subspace.sample()[None])
            sample_agent_id = torch.as_tensor(agent_id_subspace.sample()[None])
            sample_transformer_input = torch.cat([self.grid_feature_extractor(sample_grid),
                                                  self.agent_id_feature_extractor(sample_agent_id)], dim=2)
        self.transformerForStackedObservation = TransformerForStackedObservation(sample_transformer_input,
                                                                                 input_embedding_dim=cnn_output_dim * 2,
                                                                                 output_dim=transformer_output_dim)
        # Update the features dim manually
        self._features_dim = transformer_output_dim

    def forward(self, observations: TensorDict) -> torch.Tensor:

        grid_features = self.grid_feature_extractor(observations['grid'])
        agent_id_features = self.agent_id_feature_extractor(observations['agent_id_grid'])

        concat_features = torch.cat([grid_features, agent_id_features], dim=2)

        return self.transformerForStackedObservation(concat_features)
