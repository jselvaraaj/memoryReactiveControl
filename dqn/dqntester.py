from collections import defaultdict

import numpy as np
from dqn.dqnbase import DQNBase
from utils.logger import Logger
from utils.utils import vectorize_state


class DQNTester(DQNBase):
    def __init__(self, env, model, cfg):
        super().__init__(env, model)
        self.env = env
        self.model = model
        self.episode_length = int(cfg.episode_length)
        self.num_episodes = int(cfg.num_episodes)

        self.log_interval = int(cfg.log_interval)
        self.logger = Logger("Testing Runs")

    def test(self, visualize=False):

        reward_dict = defaultdict(list)
        for episode_index in range(self.num_episodes):
            print(f'\n Testing Episode {episode_index + 1}')
            self.env.reset()
            state = vectorize_state(self.env.state)
            done = False
            for step_index in range(self.episode_length):
                action = self.get_action(state)
                next_obs, reward, done, _ = self.env.step(action)
                state = vectorize_state(self.env.state)
                if visualize:
                    self.env.render()
                if done:
                    break
                reward_dict[step_index].append(reward)
                if step_index % self.log_interval == 0:
                    rewards = np.array(reward_dict[step_index])
                    mean_reward = np.mean(rewards)
                    std_reward = np.std(rewards)

                    print('Step [{:6d}/{} ({:.0f}%)]\tReward: {:.6e} ± {:.6e}'.format(
                        step_index, self.episode_length,
                        100. * step_index / self.episode_length, mean_reward, std_reward))

        self.logger.plot_and_log_scalar(reward_dict, "Reward")

        self.env.close()
