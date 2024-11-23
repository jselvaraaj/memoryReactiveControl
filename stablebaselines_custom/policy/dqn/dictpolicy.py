import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.dqn.policies import MultiInputPolicy

from stablebaselines_custom.utils.dictframestack import DictFrameStack


class StackObsMultiInputPolicy(MultiInputPolicy):
    def __init__(self, stack_last_n_obs: int,
                 observation_space: spaces.Dict, *args, **kwargs):
        super().__init__(observation_space, *args, **kwargs)
        self.stack_last_n_obs = stack_last_n_obs
        self.frames = DictFrameStack(num_frames=stack_last_n_obs, observation_space=observation_space)

    def forward(self, obs: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        self.frames.append(obs)
        return super().forward(self.frames.get(), deterministic=deterministic)
