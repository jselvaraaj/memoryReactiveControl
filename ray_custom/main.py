import hydra
from omegaconf import DictConfig

from ray_custom.configbuilders.ppogridverseconfigbuilder import PPOGridverseConfigBuilder
from ray_custom.experiment_manager import ExperimentManager


# Headless mode might work in non mac os
# import pyglet
# pyglet.options["headless"] = True


@hydra.main(version_base=None, config_path="../config/conf", config_name="config")
def main(cfg: DictConfig):
    ppo_gridverse_config_builder = PPOGridverseConfigBuilder(cfg)
    project_name = "Memory Reactive Control"
    experiment = ExperimentManager(cfg, ppo_gridverse_config_builder, project_name)

    with experiment.setup_ray():
        # experiment.train()
        experiment.sweep()


if __name__ == "__main__":
    main()
