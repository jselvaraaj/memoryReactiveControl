import os
import random
from pprint import pprint

import gymnasium as gym
import hydra
import numpy as np
import ray
import torch
from clearml import Task
from omegaconf import DictConfig
from ray.rllib.algorithms import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.utils.from_config import NotProvided
from ray.tune import register_env

from catalogs.gridversePPOcatalog import GridVersePPOCatalog
from world.worldmaker import get_gridverse_env


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    log_level = cfg.logging.log_level

    logger_config = ray.LoggingConfig(encoding="TEXT", log_level="DEBUG")

    if log_level == "ERROR":
        print("Setting logging levele to Error")
        gym.logger.set_level(gym.logger.ERROR)
        # Remember to set PYTHONWARNINGS=ignore::DeprecationWarning as env variable to control rllib warnings.
        logger_config = ray.LoggingConfig(encoding="TEXT", log_level="ERROR")

    ray.init(include_dashboard=False,
             configure_logging=True,
             log_to_driver=True,
             logging_config=logger_config
             )
    # Task.set_offline(offline_mode=True)
    task = Task.init(
        project_name="Memory Reactive Control",
        tags=["rllib", cfg.gridverse_env],
        reuse_last_task_id=False,
    )
    task.connect(cfg)

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
        )

        .reporting(log_gradients=True)
        .debugging(
            log_level="DEBUG",
            # logger_config={"type": "ray.tune.logger.NoopLogger", "logdir": "./logs"},
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                model_config=model_config, catalog_class=GridVersePPOCatalog
            )
        )
        .environment(
            "gridverse",
            env_config={
                "path": f"./gridverse_conf/{env_path}",
                "max_rollout_len": training_config.max_rollout_len,
            },
            render_env=False,
        )
        .env_runners(
            num_env_runners=env_config.num_remote_env_runners,
            num_envs_per_env_runner=env_config.num_envs_per_env_runner,
            num_cpus_per_env_runner=env_config.num_cpus_per_env_runner,
        )
        .learners(num_learners=resources_config.num_remote_learner_workers,
                  num_cpus_per_learner=resources_config.num_cpus_per_learner,
                  num_gpus_per_learner=resources_config.num_gpus_per_learner)
        .resources(
            num_cpus_for_main_process=resources_config.num_cpus_for_main_process,  # only relevant when using tune
        )
    )
    config.validate()
    print(config.to_dict())
    algo = config.build()
    task.add_tags([algo.__class__.__name__])

    for i in range(training_config.num_train_loop):
        result = algo.train()
        result.pop("config")
        pprint(result)

    model_path = os.path.abspath(os.path.join("model_registry", f"{task.id}_{algo.__class__.__name__}"))
    os.makedirs(model_path, exist_ok=True)
    checkpoint = algo.save(model_path)
    task.upload_artifact(
        name=f"rllib_{algo.__class__.__name__}", artifact_object=checkpoint
    )

    task.close()


if __name__ == "__main__":
    main()
