import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN

from ray_custom.configbuilders.ppogridverseconfigbuilder import PPOGridverseConfigBuilder
from ray_custom.experiment_manager import ExperimentManager


# Headless mode might work in non mac os
# import pyglet
# pyglet.options["headless"] = True


@hydra.main(version_base=None, config_path="../../config/hydra_conf", config_name="config")
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir
    ppo_gridverse_config_builder = PPOGridverseConfigBuilder(cfg)
    project_name = "Memory Reactive Control"
    experiment = ExperimentManager(cfg, ppo_gridverse_config_builder, project_name, output_dir)

    stop_conditions = {
        ENV_RUNNER_RESULTS: {
            EPISODE_RETURN_MEAN: 4
        },
        # f"{NUM_ENV_STEPS_SAMPLED_LIFETIME}": 2048,
        TRAINING_ITERATION: 2,
    }

    with experiment.setup_ray():
        experiment.train()
        # experiment.sweep(stop_conditions=stop_conditions)


if __name__ == "__main__":
    main()
