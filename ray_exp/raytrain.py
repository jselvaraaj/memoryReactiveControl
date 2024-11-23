import os
import random

import gymnasium as gym
import hydra
import numpy as np
import omegaconf
import ray
import torch
import wandb
from omegaconf import DictConfig
from ray.rllib.algorithms import PPOConfig
from ray.rllib.connectors.common import AddObservationsFromEpisodesToBatch, BatchIndividualItems, NumpyToTensor
from ray.rllib.connectors.learner import AddColumnsFromEpisodesToTrainBatch, AddOneTsToEpisodesAndTruncate, \
    GeneralAdvantageEstimation
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.utils.from_config import NotProvided
from ray.tune import register_env

from ray_custom.callbacks.envrendercallback import EnvRenderCallback
from ray_custom.catalogs.gridversePPOcatalog import GridVersePPOCatalog
from ray_custom.connectors.AddStatesFromEpisodesToBatchForDictSpace import AddStatesFromEpisodesToBatchForDictSpace
from world.worldmaker import get_gridverse_env


@hydra.main(version_base=None, config_path="../config/conf", config_name="config")
def main(cfg: DictConfig):
    log_level = cfg.logging.log_level
    print(f"Setting logging level to {log_level}")

    if log_level == "ERROR":
        gym.logger.set_level(gym.logger.ERROR)
        # Remember to set PYTHONWARNINGS=ignore::DeprecationWarning as env variable to control rllib warnings.

    logger_config = ray.LoggingConfig(encoding="TEXT", log_level=log_level)

    ray.init(include_dashboard=False,
             configure_logging=True,
             log_to_driver=True,
             logging_config=logger_config
             )

    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb_run = wandb.init(
        project="Memory Reactive Control",
        config=wandb_config,
        tags=["rllib", cfg.gridverse_env],
        sync_tensorboard=True
    )

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
    evaluation_config = cfg.evaluation
    train_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .framework("torch")
        .training(
            lr=training_config.learning_rate,
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
            kl_target=(
                training_config.kl_target
                if training_config.kl_target is not None
                else NotProvided
            ),
            vf_loss_coeff=training_config.vf_loss_coeff,
            entropy_coeff=training_config.entropy_coeff,
            clip_param=training_config.clip_param,
            vf_clip_param=training_config.vf_clip_param,
            add_default_connectors_to_learner_pipeline=False,
            learner_connector=lambda obs_space, action_space: [
                AddOneTsToEpisodesAndTruncate(),
                AddObservationsFromEpisodesToBatch(as_learner_connector=True),
                AddColumnsFromEpisodesToTrainBatch(),
                AddStatesFromEpisodesToBatchForDictSpace(as_learner_connector=True),
                BatchIndividualItems(multi_agent=False),
                NumpyToTensor(as_learner_connector=True, device="cpu"),
                GeneralAdvantageEstimation(
                    gamma=training_config.gamma, lambda_=training_config.lambda_
                )
            ]
        )
        .reporting(log_gradients=False)
        .debugging(
            log_level=log_level,
            logger_config={"type": "ray.tune.logger.NoopLogger", "logdir": "./logs"},
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                model_config=model_config,
                catalog_class=GridVersePPOCatalog
            )
        )
        .environment(
            # "CartPole-v1",
            "gridverse",
            env_config={
                "path": f"./gridverse_conf/{env_path}",
                "max_rollout_len": training_config.max_rollout_len,
            },
        )
        .env_runners(
            num_env_runners=env_config.num_remote_env_runners,
            num_envs_per_env_runner=env_config.num_envs_per_env_runner,
            num_cpus_per_env_runner=env_config.num_cpus_per_env_runner,
        )
        # .callbacks(
        #     WandbLoggerCallback
        # )
        .learners(num_learners=resources_config.num_remote_learner_workers,
                  num_cpus_per_learner=resources_config.num_cpus_per_learner,
                  num_gpus_per_learner=resources_config.num_gpus_per_learner)
        .resources(
            num_cpus_for_main_process=resources_config.num_cpus_for_main_process,  # only relevant when using tune
        )
    )
    train_config.validate()
    train_algo = train_config.build()
    wandb_run.tags += (train_algo.__class__.__name__,)

    # for i in range(training_config.num_train_loop):
    #     result = train_algo.train()

    model_path = os.path.abspath(os.path.join("model_registry", f"{wandb_run.id}_{train_algo.__class__.__name__}"))
    os.makedirs(model_path, exist_ok=True)
    train_algo.save(model_path)
    wandb.log_model(
        path=model_path,
        name=f"rllib_{train_algo.__class__.__name__}"
    )

    eval_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .framework("torch").reporting(log_gradients=True)
        .debugging(
            log_level=log_level,
            logger_config={"type": "ray.tune.logger.NoopLogger", "logdir": "./logs"},
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                model_config=model_config,
                catalog_class=GridVersePPOCatalog
            )
        )
        .environment(
            "gridverse",
            env_config={
                "path": f"./gridverse_conf/{env_path}",
                "max_rollout_len": evaluation_config.evaluation_config.max_rollout_len,
            },
        )
        .env_runners(
            num_env_runners=env_config.num_remote_env_runners,
            num_envs_per_env_runner=env_config.num_envs_per_env_runner,
            num_cpus_per_env_runner=env_config.num_cpus_per_env_runner,
        ).callbacks(
            EnvRenderCallback
        )
        .resources(
            num_cpus_for_main_process=resources_config.num_cpus_for_main_process,  # only relevant when using tune
        )
        .evaluation(
            evaluation_interval=evaluation_config.evaluation_interval,
            evaluation_duration=evaluation_config.evaluation_duration,
            evaluation_duration_unit=evaluation_config.evaluation_duration_unit,
            evaluation_parallel_to_training=evaluation_config.evaluation_parallel_to_training,
            evaluation_force_reset_envs_before_iteration=evaluation_config.evaluation_force_reset_envs_before_iteration,
            evaluation_num_env_runners=evaluation_config.evaluation_num_env_runners,
        ))

    eval_algo = eval_config.build()
    eval_algo.set_state(train_algo.get_state())
    result = eval_algo.evaluate()

    wandb.finish()


if __name__ == "__main__":
    main()
