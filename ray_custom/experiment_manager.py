import os
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import omegaconf
import ray
import torch
import wandb
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN
from ray.tune import register_env

from gridverse_utils.gridversemaker import get_gridverse_env
from ray_custom.callbacks.envrendercallback import EnvRenderCallback
from ray_custom.configbuilders.basegridverseconfigbuilder import BaseGridverseConfigBuilder
from ray_custom.tuner.tunerbuilder import TunerBuilder
from ray_custom.utils import get_best_checkpoint
from ray_custom.wandb_helpers.utils import upload_videos_from_result_to_wandb


class ExperimentManager:
    def __init__(self, cfg, algorithm_config_builder: BaseGridverseConfigBuilder, project_name, output_dir):
        self._cfg = cfg
        self.log_level = cfg.logging.log_level
        print(f"Setting logging level to {self.log_level}")

        if self.log_level == "ERROR":
            gym.logger.set_level(gym.logger.ERROR)
            # Remember to set PYTHONWARNINGS=ignore::DeprecationWarning as env variable to control rllib warnings.

        self.ray_logger_config = ray.LoggingConfig(encoding="TEXT", log_level=self.log_level)

        self._seed = self._cfg.seed
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)

        self.output_dir = Path(output_dir)
        self.experiment_name = self.output_dir.name
        self.best_checkpoint = None

        register_env("gridverse", get_gridverse_env)

        self.algorithm_config_builder = algorithm_config_builder
        self.ray_context = None
        self.project_name = project_name
        self.default_wandb_tags = ["rllib", self._cfg.gridverse_env]
        self.wandb_path = str(self.output_dir / "wandb")

    def setup_ray(self):
        self.ray_context = ray.init(include_dashboard=self._cfg.include_dashboard,
                                    configure_logging=True,
                                    log_to_driver=True,
                                    logging_config=self.ray_logger_config
                                    )

        return self.ray_context

    def train(self):

        wandb_config = omegaconf.OmegaConf.to_container(
            self._cfg, resolve=True, throw_on_missing=True
        )
        wandb_run = wandb.init(
            project=self.project_name,
            config=wandb_config,
            tags=self.default_wandb_tags,
            sync_tensorboard=True,
            group=self.experiment_name,
            job_type="single_train",
            dir=self.wandb_path

        )
        algorithm_config = (
            self.algorithm_config_builder.get_config().debugging(
                logger_config={"type": "ray.tune.logger.TBXLogger",
                               "logdir": str(self.output_dir / "rllib_logs")},
            ))
        algorithm = algorithm_config.build()
        wandb_run.tags += (algorithm.__class__.__name__,)

        training_config = self._cfg.hyperparameters.training
        for i in range(training_config.num_train_loop):
            algorithm.train()

        algorithm_checkpoint_registry_path = str(
            self.output_dir / "rllib_algorithm_checkpoint_registry" / f"{wandb_run.id}_{algorithm.__class__.__name__}")
        os.makedirs(algorithm_checkpoint_registry_path, exist_ok=True)
        algorithm.save(algorithm_checkpoint_registry_path)
        wandb.log_model(
            path=algorithm_checkpoint_registry_path,
            name=f"rllib_{algorithm.__class__.__name__}"
        )

        self.best_checkpoint = {"path": algorithm_checkpoint_registry_path, "wandb_run_id": wandb_run.id}

        wandb.finish()

    def sweep(self, stop_conditions, checkpoint_metric=f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}",
              checkpoint_mode="max"):
        wandb_run = wandb.init(
            project=self.project_name,
            tags=self.default_wandb_tags,
            group=self.experiment_name,
            job_type="sweep",
            dir=self.wandb_path
        )
        tuner = TunerBuilder(self.project_name, self.experiment_name, self._cfg,
                             self.algorithm_config_builder, stop_conditions).get_tuner("pbt")

        wandb_run.tags += (self.algorithm_config_builder.get_config().algo_class.__name__,)

        results = tuner.fit()

        checkpoints = results.get_best_result().best_checkpoints
        self.best_checkpoint = {"path": get_best_checkpoint(checkpoints, checkpoint_metric, checkpoint_mode).path,
                                "wandb_run_id": wandb_run.id}
        wandb.finish()

    def evaluate(self):
        if self.best_checkpoint is None:
            raise Exception("No checkpoint is available to evaluate")
        wandb_run = wandb.init(
            project=self.project_name,
            config=self.best_checkpoint,
            tags=self.default_wandb_tags,
            group=self.experiment_name,
            job_type="eval",
            dir=self.wandb_path
        )

        eval_algorithm = (
            self.algorithm_config_builder.get_config()
            .callbacks(
                EnvRenderCallback
            )
            .environment(
                env_config={
                    "max_rollout_len": self._cfg.evaluation.evaluation_config.max_rollout_len
                }
            )
        ).build()

        eval_algorithm.restore_from_path(path=self.best_checkpoint['path'])

        wandb_run.tags += (eval_algorithm.__class__.__name__,)

        result = eval_algorithm.evaluate()

        upload_videos_from_result_to_wandb(result)
        wandb.finish()
        self.best_checkpoint = None

    def __del__(self):
        self.exit_ray()

    @staticmethod
    def exit_ray():
        ray.shutdown()
