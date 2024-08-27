from gym_gridverse.gym import GymEnvironment
import gymnasium as gym
from collections import deque
import numpy as np


class GridVerseWrapper(GymEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def render(self):
        rgb_arrays = super().render()
        #Only using observations
        ret = rgb_arrays[1] if len(rgb_arrays) == 2 else rgb_arrays[0]
        return ret


class GridVerseFrameStackWrapper(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ValueError(f"Expected observation space to be of type 'Dict', but got: {type(env.observation_space)}")

        self.k = k
        self.frames = {key: deque([], maxlen=k) for key in env.observation_space.spaces.keys()}

        # Adjust the observation space to account for k stacked frames
        new_observation_space = {}
        for key, space in env.observation_space.spaces.items():
            if isinstance(space, gym.spaces.Box):
                low = np.repeat(space.low[np.newaxis, ...], k, axis=0)
                high = np.repeat(space.high[np.newaxis, ...], k, axis=0)
                new_observation_space[key] = gym.spaces.Box(low=low, high=high, dtype=space.dtype)
            else:
                raise ValueError(f"Unsupported space type for key '{key}': {type(space)}")
        self.observation_space = gym.spaces.Dict(new_observation_space)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        # Clear the frames buffer and initialize with the first observation
        for key in observation:
            for _ in range(self.k):
                self.frames[key].append(observation[key])
        return self._get_observation(), info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        # Append the new observation to the deque for each key
        for key in observation:
            self.frames[key].append(observation[key])
        return self._get_observation(), reward, done, truncated, info

    def _get_observation(self):
        # Stack frames for each key in the observation dict
        return {key: np.stack(self.frames[key], axis=0) for key in self.frames}
