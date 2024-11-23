import random

import gymnasium as gym
import numpy as np
import ray
import torch
import wandb
from omegaconf import DictConfig
from ray.rllib.algorithms import PPOConfig, Algorithm
from ray.rllib.core.rl_module import RLModuleSpec
from ray.train import Checkpoint
from ray.tune import register_env

from ray_custom.callbacks.envrendercallback import EnvRenderCallback
from ray_custom.catalogs.gridversePPOcatalog import GridVersePPOCatalog
from world.worldmaker import get_gridverse_env


def eval_checkpoint(cfg: DictConfig, checkpoint: Checkpoint, experiment_name: str):
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

    register_env("gridverse", get_gridverse_env)

    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    wandb.init(
        project="Memory Reactive Control",
        tags=["rllib", cfg.gridverse_env],
        group=experiment_name,
        job_type="eval",
        dir="../wandb"
    )

    env_path = cfg.gridverse_env
    model_config = dict(cfg.algorithm)
    env_config = cfg.environment
    resources_config = cfg.resources
    evaluation_config = cfg.evaluation

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
                "path": f"/Users/josdan/stuff/development/MemoryReactivePolicy/config/gridverse_conf/{env_path}",
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
    algo = Algorithm.from_checkpoint(path=checkpoint.path)
    eval_algo.set_state(algo.get_state())
    result = eval_algo.evaluate()

    wandb.finish()
