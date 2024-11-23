from collections import deque

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import PyTorchObs
from gymnasium import spaces

import torch as th


class DictFrameStack:
    def __init__(self, num_frames: int, observation_space: spaces.Dict):
        self.num_frames = num_frames

        self.frames = {key: deque([], maxlen=num_frames) for key in observation_space.spaces.keys()}

    def append(self, obs: PyTorchObs):
        for key in obs:
            self.frames[key].append(obs[key])

    def get(self):
        return {key: th.stack(list(self.frames[key]), dim=0) for key in self.frames}


class ClearStackedFramesCallback(BaseCallback):
    """
    Callback for clearing stacked frames at the start of every episode
    """

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.verbose >= 1:
            print(f"Num timesteps: {self.num_timesteps}")

        self.globals

        return True
