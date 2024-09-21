from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
import gymnasium as gym
from gym_gridverse.outer_env import OuterEnv
from gym_gridverse.representations.observation_representations import (
    make_observation_representation,
)
from gym_gridverse.representations.state_representations import (
    make_state_representation,
)

from world.featureextractorwrapper import GridVerseFeatureExtractorWrapper
from world.gridversewrapper import GridVerseWrapper, GridVerseFrameStackWrapper


class WorldMaker:
    @staticmethod
    def make_env(path: str, cfg) -> gym.Env:
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
        # env = GridVerseFrameStackWrapper(GridVerseWrapper(outer_env, render_mode='rgb_array'))
        env = GridVerseFeatureExtractorWrapper(GridVerseWrapper(outer_env, render_mode='rgb_array'),
                                               cfg.grid_feature_extraction.embedding_dim,
                                               cfg.grid_feature_extraction.cnn_output_dim,
                                               cfg.grid_feature_extraction.seq_model_output_dim)

        return env
