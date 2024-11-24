import gymnasium as gym
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.configs import ModelConfig

from ray_custom.encoders.gridverseencoder import GridversereEncoderConfig
from ray_custom.encoders.recurrentgridverseencoder import RecurrentGridverseEncoderConfig


class GridVersePPOCatalog(PPOCatalog):

    @classmethod
    def _get_encoder_config(cls,
                            observation_space: gym.Space,
                            model_config_dict: dict,
                            action_space: gym.Space = None,
                            ) -> ModelConfig:
        use_lstm = model_config_dict["use_lstm"]
        if use_lstm:
            tokenizer_config = cls.get_tokenizer_config(
                observation_space,
                model_config_dict,
            )
            encoder_config = RecurrentGridverseEncoderConfig(
                recurrent_layer_type="lstm",
                hidden_dim=model_config_dict["lstm_cell_size"],
                hidden_weights_initializer=model_config_dict["lstm_kernel_initializer"],
                hidden_weights_initializer_config=model_config_dict[
                    "lstm_kernel_initializer_kwargs"
                ],
                hidden_bias_initializer=model_config_dict["lstm_bias_initializer"],
                hidden_bias_initializer_config=model_config_dict[
                    "lstm_bias_initializer_kwargs"
                ],
                batch_major=True,
                num_layers=1,
                tokenizer_config=tokenizer_config,
            )
        else:
            encoder_config = GridversereEncoderConfig(observation_space=observation_space,
                                                      gridverse_encoder_config=model_config_dict['encoder'])

        return encoder_config
