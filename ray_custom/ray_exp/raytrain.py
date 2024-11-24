import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from ray_custom.configbuilders.ppogridverseconfigbuilder import PPOGridverseConfigBuilder
from ray_custom.experiment_manager import ExperimentManager


@hydra.main(version_base=None, config_path="../../config/hydra_conf", config_name="config")
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir
    ppo_gridverse_config_builder = PPOGridverseConfigBuilder(cfg)
    project_name = "Memory Reactive Control"
    experiment = ExperimentManager(cfg, ppo_gridverse_config_builder, project_name, output_dir)

    with experiment.setup_ray():
        experiment.train()
        experiment.evaluate()


if __name__ == "__main__":
    main()
