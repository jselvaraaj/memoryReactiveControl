import random
import uuid
from typing import Dict

import gymnasium as gym
import hydra
import numpy as np
import ray
import torch
import wandb
from omegaconf import DictConfig
from ray import tune, air
from ray.air.constants import TRAINING_ITERATION
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms import PPOConfig
from ray.rllib.connectors.common import AddObservationsFromEpisodesToBatch, BatchIndividualItems, NumpyToTensor
from ray.rllib.connectors.learner import AddOneTsToEpisodesAndTruncate, AddColumnsFromEpisodesToTrainBatch, \
    GeneralAdvantageEstimation
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.utils.from_config import NotProvided
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME, EPISODE_RETURN_MEAN
from ray.tune import register_env, CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.stopper import FunctionStopper

from ray_custom.catalogs.gridversePPOcatalog import GridVersePPOCatalog
from ray_custom.connectors.AddStatesFromEpisodesToBatchForDictSpace import AddStatesFromEpisodesToBatchForDictSpace
from ray_exp.rayeval import eval_checkpoint
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

    register_env("gridverse", get_gridverse_env)

    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    experiment_name = f"experiment-{uuid.uuid4().hex[:8]}"

    wandb.init(
        project="Memory Reactive Control",
        tags=["rllib", cfg.gridverse_env],
        group=experiment_name,
        job_type="sweep",
        dir="../wandb"
    )

    # Postprocess the perturbed config to ensure it's still valid
    def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size_per_learner"] < config["minibatch_size"] * 2:
            config["train_batch_size_per_learner"] = config["minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_epochs"] < 1:
            config["num_epochs"] = 1
        return config

    hyperparam_mutations = {
        "lambda": lambda: random.uniform(0.9, 1.0),
        "clip_param": lambda: random.uniform(0.01, 0.5),
        "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        "num_epochs": lambda: random.randint(1, 30),
        "minibatch_size": lambda: random.randint(128, 16384),
        "train_batch_size_per_learner": lambda: random.randint(2000, 160000),
    }

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=5,
        resample_probability=1,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations=hyperparam_mutations,
        custom_explore_fn=explore,
    )

    def stopper(trial_id: str, result: Dict):
        def execute_for_struct(a: dict, b: dict, leaf_fun):
            if isinstance(b, dict) and isinstance(a, dict):
                return any(k in a and execute_for_struct(a[k], b[k], leaf_fun) for k in b)
            return leaf_fun(a, b)

        stop_conditions = {
            ENV_RUNNER_RESULTS: {
                EPISODE_RETURN_MEAN: 4
            },
            # f"{NUM_ENV_STEPS_SAMPLED_LIFETIME}": 2048,
            TRAINING_ITERATION: 2,
        }
        if execute_for_struct(result, stop_conditions,
                              lambda result_val, stop_condition_val: result_val > stop_condition_val):
            return True
        return False

    tune_callbacks = [
        WandbLoggerCallback(
            project="Memory Reactive Control",
            upload_checkpoints=True,
            tags=["rllib", cfg.gridverse_env],
            group=experiment_name,
            job_type="train",
        )
    ]
    ppo_config = create_ppo(cfg)
    progress_reporter = CLIReporter(
        metric_columns={
            **{
                TRAINING_ITERATION: "iter",
                "time_total_s": "total time (s)",
                NUM_ENV_STEPS_SAMPLED_LIFETIME: "ts",
                f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": "combined return",
            },
            **{
                (
                    f"{ENV_RUNNER_RESULTS}/module_episode_returns_mean/" f"{pid}"
                ): f"return {pid}"
                for pid in ppo_config.policies
            },
        },
    )

    tuner = (
        tune.Tuner(
            ppo_config.algo_class,
            param_space=ppo_config,
            run_config=air.RunConfig(
                stop=FunctionStopper(stopper),
                verbose=2,
                callbacks=tune_callbacks,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_frequency=0,
                    checkpoint_at_end=True,
                ),
                progress_reporter=progress_reporter,
            ),
            tune_config=tune.TuneConfig(
                metric=f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}",
                mode="max",
                scheduler=pbt,
                num_samples=2,
                # reuse_actors=True,
            ),
        ))
    results = tuner.fit()
    wandb.finish()

    ray.shutdown()

    metric = f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}"
    mode = max
    op = max if mode == "max" else min

    def extract_value_of_checkpoint(checkpoint_tup):
        nonlocal metric
        checkpoint, checkpoint_config = checkpoint_tup
        path = metric.split('/')
        val = checkpoint_config
        for node in path:
            val = val[node]
        return val

    checkpoints = results.get_best_result().best_checkpoints
    best_checkpoint = op(checkpoints, key=extract_value_of_checkpoint)[0]

    eval_checkpoint(cfg, best_checkpoint, experiment_name)


def create_ppo(cfg):
    log_level = cfg.logging.log_level
    env_path = cfg.gridverse_env
    model_config = dict(cfg.algorithm)
    env_config = cfg.environment
    training_config = cfg.hyperparameters.training
    resources_config = cfg.resources

    return (
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
            "gridverse",
            env_config={
                "path": f"/Users/josdan/stuff/development/MemoryReactivePolicy/config/gridverse_conf/{env_path}",
                "max_rollout_len": training_config.max_rollout_len,
            }
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


if __name__ == "__main__":
    main()
