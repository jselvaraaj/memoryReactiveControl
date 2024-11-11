import gymnasium as gym
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.configs import ModelConfig

from encoders.gridverseencoder import GridversereEncoderConfig


class GridVersePPOCatalog(PPOCatalog):

    @classmethod
    def _get_encoder_config(cls,
                            observation_space: gym.Space,
                            model_config_dict: dict,
                            action_space: gym.Space = None,
                            ) -> ModelConfig:
        return GridversereEncoderConfig(observation_space=observation_space,
                                        gridverse_encoder_config=model_config_dict['encoder'])
