import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from ray.air.constants import TRAINING_ITERATION
from ray.tune.result import TIME_TOTAL_S

from ray_custom.configbuilders.ppogridverseconfigbuilder import PPOGridverseConfigBuilder
from ray_custom.experiment_manager import ExperimentManager


@hydra.main(version_base=None, config_path="../../config/hydra_conf", config_name="config")
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir
    ppo_gridverse_config_builder = PPOGridverseConfigBuilder(cfg)
    project_name = "Memory Reactive Control"
    experiment = ExperimentManager(cfg, ppo_gridverse_config_builder, project_name, output_dir)

    stop_conditions = {
        # ENV_RUNNER_RESULTS: {
        #     EPISODE_RETURN_MEAN: 4
        # },
        # f"{NUM_ENV_STEPS_SAMPLED_LIFETIME}": 2048,
        TRAINING_ITERATION: cfg.hyperparameters.training.num_train_loop,
        TIME_TOTAL_S: cfg.tuner.max_time_limit_per_trial_seconds
    }

    with experiment.setup_ray():
        experiment.sweep(stop_conditions)
        experiment.evaluate()


if __name__ == "__main__":
    main()
