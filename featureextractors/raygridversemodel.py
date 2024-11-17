from typing import override, Dict, Any

import gymnasium as gym
import torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule

from featureextractors.gridversefeatureextractor import GridVerseFeatureExtractor


# This is not used anywhere!
class RayGridVerseModule(PPOTorchRLModule):
    def __init__(self, observation_space: gym.spaces.Dict, action_space: gym.spaces.Discrete, model_config: Dict, *args,
                 **kwargs):
        super().__init__(observation_space=observation_space, action_space=action_space, model_config=model_config,
                         *args, **kwargs)

        self.gridverse_feature_extractor = GridVerseFeatureExtractor(observation_space, model_config)

        self.combined_feature_size = model_config['grid_encoder']['output_dim'] + model_config['agent_id_encoder'][
            'output_dim'] + \
                                     model_config['items_encoder']['layers'][-1]

        self.policy_head = torch.nn.Sequential(torch.nn.Linear(self.combined_feature_size, action_space.n),
                                               torch.nn.Softmax(dim=1))
        self.value_head = torch.nn.Linear(self.combined_feature_size, 1)

    @override
    def _forward(self, batch: Dict[str, Any], **kwargs):
        observations = batch['obs']
        concat_features = self.gridverse_feature_extractor(observations)

        output = self.policy_head(concat_features)
        self.value = self.value_head(concat_features).squeeze(1)

        return {"action_dist_inputs": output}

    def value_function(self):
        assert self.value is not None, "must call forward first!"
        return self.value
