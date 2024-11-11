import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict

from featureextractors.gridversefeatureextractor import GridVerseFeatureExtractor


class GridVerseFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.vector.utils.spaces.Dict,
            config: dict,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        self.gridverse_feature_extractor = GridVerseFeatureExtractor(observation_space, config)

        # Update the features dim manually
        self._features_dim = config['grid_encoder']['output_dim'] + config['agent_id_encoder']['output_dim'] + \
                             config['items_encoder']['layers'][-1]

    def forward(self, observations: TensorDict) -> torch.Tensor:
        concat_features = self.gridverse_feature_extractor(observations)
        return concat_features
