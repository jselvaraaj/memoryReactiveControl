import torch


def setup(task, cfg):
    torch.set_printoptions(precision=4, sci_mode=False)

    task.connect(cfg)
    torch.manual_seed(cfg.seed)

    # configs
    algorithm_config = cfg.algorithm
    hyperparams_config = cfg.hyperparameters
    environment_config = cfg.environment
    logging_config = cfg.logging

    print("Environment config:", environment_config, "\n")
    print("Algorithm config:", algorithm_config, "\n")
    print("Hyperparameters:", hyperparams_config, "\n")
    print("Logging config:", logging_config, "\n")

    return algorithm_config, hyperparams_config, environment_config, logging_config
