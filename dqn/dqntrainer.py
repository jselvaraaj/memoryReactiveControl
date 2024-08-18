import torch

from dqn.dqnbase import DQNBase
from dqn.utils.replaybuffer import ReplayBuffer
from utils.logger import Logger
from utils.utils import vectorize_state, get_dtype
from collections import defaultdict
import numpy as np


class DQNTrainer(DQNBase):
    # epsilon is the probability of choosing a random action. for exploration
    def __init__(self, env, model, cfg, device):
        super().__init__(env, model, device)
        self.dtype = get_dtype(cfg.dtype)

        self.replayBuffer = ReplayBuffer(int(cfg.replay_buffer_memory))
        self.epsilon = cfg.policy_epsilon
        self.batch_size = cfg.batch_size
        self.discount_factor = cfg.discount_factor
        self.episode_length = int(cfg.episode_length)
        self.num_episodes = int(cfg.num_episodes)

        self.step_log_interval = int(cfg.step_log_interval)

        self.state_dim = vectorize_state(self.env.state, self.dtype).shape[0]
        self.logger = Logger("training_runs")

    def get_action(self, state: torch.Tensor) -> int:
        if torch.rand(1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return super().get_action(state)

    def get_batch(self, batch_size):
        batch = torch.stack(self.replayBuffer.sample(batch_size))
        # batch = batch.refine_names('batch_dim', 'state_dim_action_dim_reward_dim_next_state_dim_done_dim')
        return batch

    def vectorize_sars(self, sars_with_done_flag: (torch.Tensor, int, float, torch.Tensor, bool)) -> torch.Tensor:
        (state, action, reward, next_state, done) = sars_with_done_flag
        vectorized_sars = torch.cat([state[None, :],
                                     torch.tensor(action, dtype=self.dtype)[None, None],
                                     torch.tensor(reward, dtype=self.dtype)[None, None],
                                     next_state[None, :],
                                     torch.tensor(1 if done else 0, dtype=self.dtype)[None, None]], dim=-1)[0]
        # removing the extra dimension created in the front to concat
        # vectorized_sars.refine_names('state_dim_action_dim_reward_dim_next_state_dim_done_dim')
        return vectorized_sars

    def de_vectorize_sars_from_batch(self, batch) -> (
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        states = batch[:, :self.state_dim]
        actions = batch[:, self.state_dim]
        rewards = batch[:, self.state_dim + 1]
        next_states = batch[:, self.state_dim + 2:-1]
        dones = batch[:, -1]

        # states.rename(state_dim_action_dim_reward_dim_next_state_dim_done_dim='state_dim')
        # actions.refine_names('batch_dim')
        # rewards.refine_names('batch_dim')
        # next_states.rename(state_dim_action_dim_reward_dim_next_state_dim_done_dim='state_dim')
        # dones.refine_names('batch_dim')

        return states.to(device=self.device), actions.to(device=self.device), rewards.to(
            device=self.device), next_states.to(
            device=self.device), dones.to(device=self.device)

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_dict = defaultdict(list)
        self.model.train()

        for episode_index in range(self.num_episodes):
            print(f'\nEpisode {episode_index + 1}')
            self.env.reset()
            state = vectorize_state(self.env.state, self.dtype)
            done = False
            for step_index in range(self.episode_length):

                # Execute policy and collect data to add to replay buffer
                action = self.get_action(state)
                next_obs, reward, done, _ = self.env.step(action)
                next_state = vectorize_state(self.env.state, self.dtype)
                self.replayBuffer.add(self.vectorize_sars((state, action, reward, next_state, done)))

                # Sample from replay buffer and calculate loss
                batch = self.get_batch(self.batch_size)
                states, actions, rewards, next_states, dones = self.de_vectorize_sars_from_batch(batch)
                discount_factor = torch.Tensor(
                    [self.discount_factor]).to(dtype=self.dtype, device=self.device)
                y = rewards * dones + discount_factor * self.model(next_states).max(dim=1)[0] * (1 - dones)
                # Using int64 here because the gather function requires it
                # Dropping name here because gather and MSLoss functions does not support named tensors yet :((((
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
                if step_index % self.step_log_interval == 0:
                    loss_dict[step_index].append(loss.data.item())
                    losses = np.array(loss_dict[step_index])
                    mean_loss = np.mean(losses)
                    std_loss = np.std(losses)

                    print('Step [{:6d}/{} ({:.0f}%)]\tLoss: {:.6e} ± {:.6e}\t Action: '.format(
                        step_index, self.episode_length,
                        100. * step_index / self.episode_length, mean_loss, std_loss),
                        self.env.outer_env.action_space.int_to_action(action), '\nQ Value: ',
                        self.get_q_values(state).detach())
                    print()

        self.logger.plot_and_log_scalar(loss_dict, "Loss")
        self.logger.writer.flush()
