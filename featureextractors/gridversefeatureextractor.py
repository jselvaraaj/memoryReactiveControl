import torch
import gymnasium as gym
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from featureextractors.agentidfeatureextractor import AgentIdFeatureExtractor
from featureextractors.rnnencoder import StatefulRNN
from featureextractors.stacked.agentidfeatureextractor import AgentIdStackedFeatureExtractor
from featureextractors.gridfeatureextractor import GridFeatureExtractor
from featureextractors.stacked.gridframefeatureextractor import GridFrameStackedFeatureExtractor
from featureextractors.stacked.rnnencoder import RNNForStackedFeatures
from featureextractors.stacked.transformerencoder import TransformerForStackedFeatures


class GridVerseFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.vector.utils.spaces.Dict,
            grid_embedding_dim: int,
            grid_cnn_config=None,
            agent_id_cnn_config=None,
            cnn_output_dim: int = 128,
            seq_model_output_dim: int = 128,
            seq_model_hidden_dim: int = 64,
            seq_model_num_layers: int = 1,
            seq_model_type: str = 'rnn',
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        self.is_stacked_frame = False

        # Grid
        grid_subspace = observation_space.spaces['grid']
        if not isinstance(grid_subspace, gym.spaces.Box):
            raise ValueError(f"Expected observation space to be of type 'Box', but got: {type(grid_subspace)}")
        # The i,j is grid co-ordinates. the last index is the grid object index type; for now we are ignoring the color and items in the grid
        number_of_objects = grid_subspace.high.flatten()[0] - grid_subspace.low.flatten()[0] + 1
        ndim = grid_subspace.high.ndim
        # ndim = 3 means(x, y, [object,color,item]])
        # ndim = 4 means(seq_len, x, y, [object,color,item]])
        if ndim == 3:
            self.grid_feature_extractor = GridFeatureExtractor(number_of_objects, grid_embedding_dim, cnn_output_dim,
                                                               grid_subspace, grid_cnn_config)
        elif ndim == 4:
            self.is_stacked_frame = True
            self.grid_feature_extractor = GridFrameStackedFeatureExtractor(grid_subspace, number_of_objects,
                                                                           grid_embedding_dim,
                                                                           cnn_output_dim)
        else:
            raise ValueError(f"Unsupported grid tensor shape: {grid_subspace.shape}")

        # Agent ID
        agent_id_subspace = observation_space.spaces['agent_id_grid']

        if self.is_stacked_frame:
            self.agent_id_feature_extractor = AgentIdStackedFeatureExtractor(agent_id_subspace, cnn_output_dim)
        else:
            self.agent_id_feature_extractor = AgentIdFeatureExtractor(agent_id_subspace, cnn_output_dim,
                                                                      agent_id_cnn_config)

        with torch.no_grad():
            sample_grid = torch.as_tensor(grid_subspace.sample()[None])
            sample_agent_id = torch.as_tensor(agent_id_subspace.sample()[None])
            seq_model_input = self.concat_features(self.grid_feature_extractor(sample_grid),
                                                   self.agent_id_feature_extractor(sample_agent_id))
        self.seq_model_type = seq_model_type.lower()
        if self.is_stacked_frame:
            # Assumes [batch, seq_len ,embedding_dim]
            if self.seq_model_type == 'rnn':
                self.seq_model = RNNForStackedFeatures(seq_model_input,
                                                       input_embedding_dim=cnn_output_dim * 2,
                                                       output_dim=seq_model_output_dim)
            elif self.seq_model_type == 'transformer':
                self.seq_model = TransformerForStackedFeatures(seq_model_input,
                                                               input_embedding_dim=cnn_output_dim * 2,
                                                               output_dim=seq_model_output_dim)
            else:
                raise ValueError(f"Unsupported sequence model type: {self.seq_model_type}")
        else:
            # Assumes [batch ,embedding_dim]
            if self.seq_model_type == 'rnn':
                self.seq_model = StatefulRNN(seq_model_input,
                                             input_embedding_dim=cnn_output_dim * 2,
                                             hidden_dim=seq_model_hidden_dim,
                                             num_layers=seq_model_num_layers,
                                             output_dim=seq_model_output_dim)
            else:
                raise ValueError(f"Unsupported sequence model type: {self.seq_model_type}")

        # Update the features dim manually
        self._features_dim = seq_model_output_dim

    def concat_features(self, grid_features: torch.Tensor, agent_id_feature: torch.Tensor) -> torch.Tensor:
        if self.is_stacked_frame:
            # concat_features dim is [batch, seq_len, cnn_output_dim * 2]
            return torch.cat([grid_features, agent_id_feature], dim=2)
        else:
            # concat_features dim is [batch, cnn_output_dim * 2]
            return torch.cat([grid_features, agent_id_feature], dim=1)

    def forward(self, observations: TensorDict, hidden_state: torch.Tensor = None,
                cell_state: torch.Tensor = None) -> torch.Tensor:

        grid_features = self.grid_feature_extractor(observations['grid'])
        agent_id_features = self.agent_id_feature_extractor(observations['agent_id_grid'])

        concat_features = self.concat_features(grid_features, agent_id_features)

        # Returns [batch, seq_model_output_dim]
        if hidden_state is None or cell_state is None:
            return self.seq_model(concat_features)
        return self.seq_model(concat_features, hidden_state, cell_state)
