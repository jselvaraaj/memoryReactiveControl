from ray.rllib.algorithms import PPOConfig
from ray.rllib.utils.from_config import NotProvided

from ray_custom.configbuilders.basegridverseconfigbuilder import BaseGridverseConfigBuilder


class PPOGridverseConfigBuilder(BaseGridverseConfigBuilder):

    def __init__(self, cfg):
        super().__init__(cfg=cfg, algorithm_config_ctor=PPOConfig)
        self._setup_ppo()

    def _setup_ppo(self):
        training_config = self._cfg.hyperparameters.training
        self.algorithm_config = (
            self.algorithm_config
            .training(
                use_critic=training_config.use_critic,
                use_gae=training_config.use_gae,
                lambda_=training_config.lambda_,
                use_kl_loss=training_config.use_kl_loss,
                kl_coeff=training_config.kl_coeff,
                kl_target=(
                    training_config.kl_target
                    if training_config.kl_target is not None
                    else NotProvided
                ),
                vf_loss_coeff=training_config.vf_loss_coeff,
                entropy_coeff=training_config.entropy_coeff,
                clip_param=training_config.clip_param,
                vf_clip_param=training_config.vf_clip_param,
            )
        )

    def get_config(self):
        return PPOConfig.from_state(self.algorithm_config.get_state())
