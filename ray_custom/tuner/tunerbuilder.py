from ray import tune, air
from ray.air.constants import TRAINING_ITERATION
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN, NUM_ENV_STEPS_SAMPLED_LIFETIME
from ray.tune import CLIReporter
from ray.tune.stopper import FunctionStopper

from ray_custom.configbuilders.basegridverseconfigbuilder import BaseGridverseConfigBuilder
from ray_custom.tuner.scheduler.populationbasedtrainingbuilder import PopulationBasedTrainingBuilder
from ray_custom.tuner.utils import get_tune_trial_stopper


class TunerBuilder:

    def __init__(self, project_name, experiment_name, cfg, algorithm_config_builder: BaseGridverseConfigBuilder,
                 stop_conditions):
        self.stop_conditions = stop_conditions
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.cfg = cfg
        self.algorithm_config_builder = algorithm_config_builder
        self.log_level = 1 if self.cfg.logging.log_level != "DEBUG" else 2

        self.callbacks = [
            WandbLoggerCallback(
                project=self.project_name,
                upload_checkpoints=True,
                tags=["rllib", cfg.gridverse_env],
                group=self.experiment_name,
                job_type="sweep-trials",
            )
        ]

        self.progress_reporter = CLIReporter(
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
                    for pid in self.algorithm_config_builder.get_config().policies
                },
            },
        )

    def get_tuner(self, scheduler_type: str = "pbt"):
        algorithm_config = self.algorithm_config_builder.get_config()
        stopper = get_tune_trial_stopper(self.stop_conditions)
        tuner_config = self.cfg.tuner

        if scheduler_type != "pbt":
            raise Exception(f"Not implemented scheduler for ${scheduler_type}")

        scheduler = PopulationBasedTrainingBuilder(tuner_config.population_based_training).build()

        return tune.Tuner(
            algorithm_config.algo_class,
            param_space=algorithm_config,
            run_config=air.RunConfig(
                stop=FunctionStopper(stopper),
                verbose=self.log_level,
                callbacks=self.callbacks,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_frequency=tuner_config.checkpoint_frequency,
                    checkpoint_at_end=tuner_config.checkpoint_at_end,
                ),
                progress_reporter=self.progress_reporter,
            ),
            tune_config=tune.TuneConfig(
                metric=f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}",
                mode="max",
                scheduler=scheduler,
                num_samples=tuner_config.num_samples,
                # reuse_actors=True, # PPO doesn't implement reset_config
            ),
        )
