import gymnasium as gym
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.outer_env import OuterEnv
from gym_gridverse.representations.observation_representations import (
    make_observation_representation,
)
from gym_gridverse.representations.state_representations import (
    make_state_representation,
)
from gymnasium.wrappers.time_limit import TimeLimit

from gridverse_utils.gridversewrapper import GridVerseWrapper


class WorldMaker:
    def __init__(self, path: str):
        self.path = path
        print(f'Loading gridverse using YAML in {self.path}')
        self.inner_env = factory_env_from_yaml(self.path)
        self.state_representation = make_state_representation(
            'default',
            self.inner_env.state_space,
        )
        self.observation_representation = make_observation_representation(
            'default',
            self.inner_env.observation_space,
        )
        self.outer_env = OuterEnv(
            self.inner_env,
            state_representation=self.state_representation,
            observation_representation=self.observation_representation,
        )

        self.env_config = {'outer_env': self.outer_env, 'render_mode': 'rgb_array'}

    def make_env(self) -> gym.Env:
        env = GridVerseWrapper(self.env_config)
        return env

    def get_env_with_args(self):
        return GridVerseWrapper, self.env_config


def get_gridverse_env(config: dict):
    path = config['path']
    max_rollout_len = config['max_rollout_len']
    env_maker = WorldMaker(path)
    return TimeLimit(env_maker.make_env(), max_episode_steps=max_rollout_len)
