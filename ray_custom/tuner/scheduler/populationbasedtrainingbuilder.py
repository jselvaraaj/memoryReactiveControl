import random

from ray.tune.schedulers import PopulationBasedTraining


class PopulationBasedTrainingBuilder:
    def __init__(self, tuner_config):
        self.tuner_config = tuner_config

        self.hyperparam_mutations = {
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.01, 0.5),
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "num_epochs": lambda: random.randint(1, 30),
            "minibatch_size": lambda: random.randint(128, 16384),
            "train_batch_size_per_learner": lambda: random.randint(1024, 2 ^ 12),
            "entropy_coeff": lambda: random.uniform(0.001, 5 * 0.001),
            "clip_param": lambda: random.uniform(0.2, 0.8),
        }

    def build(self):
        return PopulationBasedTraining(
            time_attr=self.tuner_config.time_attr,
            perturbation_interval=self.tuner_config.perturbation_interval,
            resample_probability=self.tuner_config.resample_probability,
            # Specifies the mutations of these hyperparams
            hyperparam_mutations=self.hyperparam_mutations,
            custom_explore_fn=explore,
        )


def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size_per_learner"] < config["minibatch_size"] * 2:
        config["train_batch_size_per_learner"] = config["minibatch_size"] * 2
    # ensure we run at least one sgd iter
    if config["num_epochs"] < 1:
        config["num_epochs"] = 1
    return config
