import gymnasium as gym
import numpy as np
import torch

from featureextractors.gridversefeatureextractor import GridVerseFeatureExtractor


def fix_obs(observation):
    observation['grid'] = torch.tensor(observation['grid'][None])
    observation['agent_id_grid'] = torch.tensor(observation['agent_id_grid'][None])

    return observation


class GridVerseFeatureExtractorWrapper(gym.Wrapper):
    def __init__(self, env, grid_embedding_dim, cnn_output_dim, seq_model_output_dim, seq_model_type='rnn'):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ValueError(f"Expected observation space to be of type 'Dict', but got: {type(env.observation_space)}")

        self.extractor = GridVerseFeatureExtractor(env.observation_space, grid_embedding_dim,
                                                   cnn_output_dim,
                                                   seq_model_output_dim,
                                                   seq_model_type)
        self.hidden_state = None
        self.cell_state = None

        feature_dim = seq_model_output_dim
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.hidden_state = None
        self.cell_state = None
        observation = fix_obs(observation)
        features, self.hidden_state, self.cell_state = self.extractor(observation, self.hidden_state, self.cell_state)

        return features.detach().numpy(), info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        observation = fix_obs(observation)

        # Extract features
        features, self.hidden_state, self.cell_state = self.extractor(observation, self.hidden_state, self.cell_state)

        return features.detach().numpy(), reward, done, truncated, info
