import random
from pprint import pprint

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from ray.rllib.algorithms import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.utils.from_config import NotProvided
from ray.tune import register_env

from catalogs.gridversePPOcatalog import GridVersePPOCatalog
from world.worldmaker import get_gridverse_env


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    register_env("gridverse", get_gridverse_env)

    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_path = cfg.gridverse_env
    model_config = dict(cfg.algorithm)
    training_config = cfg.hyperparameters.training
    env_config = cfg.environment
    resources_config = cfg.resources

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .framework("torch")
        .training(lr=training_config.learning_rate,
                  gamma=training_config.gamma,
                  train_batch_size_per_learner=training_config.train_batch_size_per_learner,
                  num_epochs=training_config.num_epochs,
                  minibatch_size=training_config.minibatch_size,
                  shuffle_batch_per_epoch=training_config.shuffle_batch_per_epoch,
                  grad_clip=training_config.grad_clip,
                  grad_clip_by=training_config.grad_clip_by,
                  use_critic=training_config.use_critic,
                  use_gae=training_config.use_gae,
                  lambda_=training_config.lambda_,
                  use_kl_loss=training_config.use_kl_loss,
                  kl_coeff=training_config.kl_coeff,
                  kl_target=training_config.kl_target if training_config.kl_target is not None else NotProvided,
                  vf_loss_coeff=training_config.vf_loss_coeff,
                  entropy_coeff=training_config.entropy_coeff,
                  clip_param=training_config.clip_param,
                  vf_clip_param=training_config.vf_clip_param
                  )
        .resources(num_learner_workers=resources_config.num_learner_workers,
                   num_cpus_for_main_process=resources_config.num_cpus_for_main_process,
                   num_gpus=resources_config.num_gpus)
        .rl_module(
            rl_module_spec=RLModuleSpec(model_config=model_config, catalog_class=GridVersePPOCatalog)
        )
        .environment("gridverse", env_config={
            "path": f'./gridverse_conf/{env_path}',
            "max_rollout_len": training_config.max_rollout_len})
        .env_runners(num_env_runners=env_config.num_env_runners,
                     num_envs_per_env_runner=env_config.num_envs_per_env_runner,
                     num_cpus_per_env_runner=env_config.num_cpus_per_env_runner)
    )

    algo = config.build()

    for i in range(training_config.num_train_loop):
        result = algo.train()
        result.pop("config")
        pprint(result)


main()
