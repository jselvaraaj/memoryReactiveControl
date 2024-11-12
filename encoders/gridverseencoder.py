from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gymnasium as gym
import ray.rllib.core.models.base as ray_models
from ray.rllib.core import Columns
from ray.rllib.core.models.base import Encoder
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.torch.base import TorchModel

from featureextractors.gridversefeatureextractor import GridVerseFeatureExtractor


@dataclass
class GridversereEncoderConfig(ModelConfig):
    observation_space: gym.Space = None
    gridverse_encoder_config: Dict = None

    def build(self, framework: str = "torch"):
        if framework != "torch":
            raise Exception("GridverseEncoder is only available in pytorch")
        if self.observation_space is None:
            raise Exception(f"observation_space is None for {self.__class__}", )
        if self.gridverse_encoder_config is None:
            raise Exception(f"gridverse_encoder_config is None for {self.__class__}", )

        return GridversereEncoder(self)

    @property
    def output_dims(self) -> Optional[Tuple[int]]:
        output_dim = self.gridverse_encoder_config['grid_encoder']['output_dim'] + \
                     self.gridverse_encoder_config['agent_id_encoder'][
                         'output_dim'] + \
                     self.gridverse_encoder_config['items_encoder']['layers'][-1]

        return (output_dim,)


class GridversereEncoder(TorchModel, Encoder):

    def __init__(self, config: GridversereEncoderConfig) -> None:
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        if not isinstance(config.observation_space, gym.spaces.Dict):
            raise Exception(f"GridversereEncoderConfig.observation_space is not a dict space in {self.__class__}", )

        self.encoder = GridVerseFeatureExtractor(config.observation_space, config.gridverse_encoder_config)

        # self._initial_weights = None  # Placeholder for initial weights
        # self._checked_weights = False

    def _forward(self, input_dict: dict, **kwargs) -> dict:
        # # Check weights on the first forward pass if not already checked
        # if not self._checked_weights:
        #     self.store_initial_weights()
        #     self._checked_weights = True
        # else:
        #     self.are_weights_updated()

        observations = input_dict[Columns.OBS]
        output = self.encoder(observations)
        outputs = {ray_models.ENCODER_OUT: output}
        return outputs

    # def store_initial_weights(self):
    #     """Store the initial weights for later comparison."""
    #     self._initial_weights = {name: param.clone() for name, param in self.named_parameters()}
    #
    # def are_weights_updated(self):
    #     """Check if weights have been updated compared to the initial weights."""
    #     if self._initial_weights is None:
    #         raise ValueError("Initial weights have not been stored.")
    #
    #     updates_found = False
    #     for name, param in self.named_parameters():
    #         if not torch.equal(self._initial_weights[name], param):
    #             updates_found = True
    #             break
    #     if not updates_found:
    #         print("No weights have been updated.")
    #     else:
    #         print(f"Weights for have been updated.")
    #
    #     self.store_initial_weights()
