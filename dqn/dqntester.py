from collections import defaultdict

import numpy as np
import torch

from dqn.dqnbase import DQNBase
from utils.logger import Logger
from utils.utils import vectorize_state


class DQNTester(DQNBase):
    def __init__(self, env, model, cfg):
        super().__init__(env, model)
        self.env = env
        self.model = model
        self.episode_length = int(cfg.episode_length)
        self.num_episodes = int(cfg.test_num_episodes)

        self.step_log_interval = int(cfg.step_log_interval)
        self.episode_log_interval = int(cfg.episode_log_interval)
        self.logger = Logger("testing_runs")

    def test(self, visualize=False):
        self.model.eval()
        reward_dict = defaultdict(list)
        for episode_index in range(self.num_episodes):
            print(f'\n Testing Episode {episode_index + 1}')
            self.env.reset()
            state = vectorize_state(self.env.state)
            done = False
            episode_frames = []
            for step_index in range(self.episode_length):
                action = self.get_action(state)
                next_obs, reward, done, _ = self.env.step(action)
                state = vectorize_state(self.env.state)
                if done:
                    break
                reward_dict[step_index].append(reward)
                if step_index % self.step_log_interval == 0:
                    rewards = np.array(reward_dict[step_index])
                    mean_reward = np.mean(rewards)
                    std_reward = np.std(rewards)

                    print('Step [{:6d}/{} ({:.0f}%)]\tReward: {:.6e} ± {:.6e}\tAction: '.format(
                        step_index, self.episode_length,
                        100. * step_index / self.episode_length, mean_reward, std_reward),
                        self.env.outer_env.action_space.int_to_action(action))
                if episode_index % self.episode_log_interval == 0:
                    episode_frames.append(torch.from_numpy(self.env.render(mode='rgb_array_state').copy()))

            if episode_index % self.episode_log_interval == 0:
                video_tensor = torch.stack(episode_frames, dim=0).permute(0, 3, 1, 2)[None, :]
                self.logger.writer.add_video(f"Episode {episode_index + 1} State",
                                             video_tensor, global_step=episode_index + 1, fps=0.25)

        self.logger.plot_and_log_scalar(reward_dict, "Reward")
        self.logger.writer.flush()
