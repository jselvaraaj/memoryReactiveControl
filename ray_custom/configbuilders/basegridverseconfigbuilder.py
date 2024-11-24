from pathlib import Path

from ray.rllib.connectors.common import AddObservationsFromEpisodesToBatch, BatchIndividualItems, NumpyToTensor
from ray.rllib.connectors.learner import AddOneTsToEpisodesAndTruncate, AddColumnsFromEpisodesToTrainBatch, \
    GeneralAdvantageEstimation
from ray.rllib.core.rl_module import RLModuleSpec

from ray_custom.catalogs.gridversePPOcatalog import GridVersePPOCatalog
from ray_custom.connectors.AddStatesFromEpisodesToBatchForDictSpace import AddStatesFromEpisodesToBatchForDictSpace


class BaseGridverseConfigBuilder:
    def __init__(self, cfg, algorithm_config_ctor):
        self._cfg = cfg
        self.algorithm_config = algorithm_config_ctor()
        self._setup_base()

    def _setup_base(self):
        log_level = self._cfg.logging.log_level
        env_path = self._cfg.gridverse_env
        model_config = dict(self._cfg.algorithm)
        env_config = self._cfg.environment
        training_config = self._cfg.hyperparameters.training
        resources_config = self._cfg.resources
        evaluation_config = self._cfg.evaluation
        current_dir = Path(__file__).parent
        self.algorithm_config = (
            self.algorithm_config
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .framework("torch")
            .reporting(log_gradients=False)
            .debugging(
                log_level=log_level,
                logger_config={"type": "ray.tune.logger.NoopLogger"},
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
                    "path": str(current_dir / ".." / ".." / "config" / "gridverse_conf" / env_path),
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
            ).evaluation(
                evaluation_interval=evaluation_config.evaluation_interval,
                evaluation_duration=evaluation_config.evaluation_duration,
                evaluation_duration_unit=evaluation_config.evaluation_duration_unit,
                evaluation_parallel_to_training=evaluation_config.evaluation_parallel_to_training,
                evaluation_force_reset_envs_before_iteration=evaluation_config.evaluation_force_reset_envs_before_iteration,
                evaluation_num_env_runners=evaluation_config.evaluation_num_env_runners,
            ).training(
                lr=training_config.learning_rate,
                gamma=training_config.gamma,
                train_batch_size_per_learner=training_config.train_batch_size_per_learner,
                num_epochs=training_config.num_epochs,
                minibatch_size=training_config.minibatch_size,
                shuffle_batch_per_epoch=training_config.shuffle_batch_per_epoch,
                grad_clip=training_config.grad_clip,
                grad_clip_by=training_config.grad_clip_by,
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
                ])
        )

    def get_config(self):
        return self.algorithm_config.copy()
