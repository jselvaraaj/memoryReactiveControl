from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.gym import GymEnvironment
from gym_gridverse.outer_env import OuterEnv
from gym_gridverse.representations.observation_representations import (
    make_observation_representation,
)
from gym_gridverse.representations.state_representations import (
    make_state_representation,
)
from gym.wrappers import FlattenObservation


class WorldMaker:
    @staticmethod
    def make_env(path: str) -> GymEnvironment:
        print(f'Loading using YAML in {path}')
        inner_env = factory_env_from_yaml(path)
        state_representation = make_state_representation(
            'default',
            inner_env.state_space,
        )
        observation_representation = make_observation_representation(
            'default',
            inner_env.observation_space,
        )
        outer_env = OuterEnv(
            inner_env,
            state_representation=state_representation,
            observation_representation=observation_representation,
        )
        env = GymEnvironment(outer_env)

        return env
