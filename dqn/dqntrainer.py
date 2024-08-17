import torch

from dqn.dqnbase import DQNBase
from dqn.utils.replaybuffer import ReplayBuffer
from utils.logger import Logger
from utils.utils import vectorize_state
from collections import defaultdict
import numpy as np


class DQNTrainer(DQNBase):

    # epsilon is the probability of choosing a random action. for exploration
    def __init__(self, env, model, cfg):
        super().__init__(env, model)

        self.replayBuffer = ReplayBuffer(int(cfg.replay_buffer_memory))
        self.epsilon = cfg.policy_epsilon
        self.batch_size = cfg.batch_size
        self.discount_factor = cfg.discount_factor
        self.episode_length = int(cfg.episode_length)
        self.num_episodes = int(cfg.num_episodes)

        self.log_interval = int(cfg.log_interval)

        self.state_dim = vectorize_state(self.env.state).shape[0]
        self.logger = Logger("Training Runs")

    def get_action(self, state):
        if torch.rand(1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return super().get_action(state)

    def get_batch(self, batch_size):
        batch = self.replayBuffer.sample(batch_size)
        return torch.stack(batch).float()

    @staticmethod
    def vectorize_sars(sars_with_done_flag):
        (state, action, reward, next_state, done) = sars_with_done_flag
        return torch.cat([state[None, :],
                          torch.tensor(action)[None, None],
                          torch.tensor(reward)[None, None],
                          next_state[None, :],
                          torch.tensor(1 if done else 0)[None, None]], dim=-1)[0]
        # removing the extra dimension created in the front to concat

    def de_vectorize_sars_from_batch(self, batch):

        state = batch[:, :self.state_dim]
        action = batch[:, self.state_dim]
        reward = batch[:, self.state_dim + 1]
        next_state = batch[:, self.state_dim + 2:-1]
        done = batch[:, -1]

        return state, action, reward, next_state, done

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_dict = defaultdict(list)

        for episode_index in range(self.num_episodes):
            print(f'\nEpisode {episode_index + 1}')
            self.env.reset()
            state = vectorize_state(self.env.state)
            done = False
            for step_index in range(self.episode_length):

                # Execute policy and collect data to add to replay buffer
                action = self.get_action(state)
                next_obs, reward, done, _ = self.env.step(action)
                next_state = vectorize_state(self.env.state)
                self.replayBuffer.add(self.vectorize_sars((state, action, reward, next_state, done)))

                # Sample from replay buffer and calculate loss
                batch = self.get_batch(self.batch_size)
                states, actions, rewards, next_states, dones = self.de_vectorize_sars_from_batch(batch)
                y = rewards * dones + self.discount_factor * self.model(next_states).max(dim=1)[0] * (1 - dones)
                prediction = self.model(states).gather(1, actions.to(torch.int64)[:, None]) \
                    [:, 0]  # removing the new dimension created by none
                loss = torch.nn.MSELoss()(prediction, y)

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                state = next_state
                if done:
                    break
                if step_index % self.log_interval == 0:
                    loss_dict[step_index].append(loss.data.item())
                    losses = np.array(loss_dict[step_index])
                    mean_loss = np.mean(losses)
                    std_loss = np.std(losses)

                    print('Step [{:6d}/{} ({:.0f}%)]\tLoss: {:.6e} ± {:.6e}'.format(
                        step_index, self.episode_length,
                        100. * step_index / self.episode_length, mean_loss, std_loss))

        self.logger.plot_and_log_scalar(loss_dict, "Loss")
