import collections

import torch
from utils.replaybuffer import ReplayBuffer


class DQNTrainer:

    # epsilon is the probability of choosing a random action. for exploration
    def __init__(self, env, model, replay_buffer_memory=1000, epsilon=0.1, batch_size=32, discount_factor=0.99):
        self.env = env
        self.model = model
        self.replayBuffer = ReplayBuffer(replay_buffer_memory)
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.discount_factor = discount_factor

    def get_action(self, state):
        if torch.rand(1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return torch.argmax(self.model(torch.tensor(state).float()))

    def get_batch(self, batch_size):
        return torch.tensor(self.replayBuffer.sample(batch_size)).float()

    def vectorize_sars(self, sars_with_done_flag):
        (state, action, reward, next_state, done) = sars_with_done_flag
        return torch.stack([torch.tensor(state).float(),
                            action, reward,
                            torch.tensor(next_state).float(),
                            torch.tensor(1 if done else 0)])

    def devectorize_sars_from_batch(self, batch):
        pass

    def train(self, episode_length=1000, num_episodes=1000):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        for i in range(num_episodes):
            state = self.env.reset()
            done = False
            for _ in range(episode_length):

                # Execute policy and collect data to add to replay buffer
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.replayBuffer.add(self.vectorize_sars((state, action, reward, next_state, done)))

                # Sample from replay buffer and calculate loss
                batch = self.get_batch(self.batch_size)
                states, actions, rewards, next_states, dones = self.devectorize_sars_from_batch(batch)
                y = rewards * dones + self.discount_factor * self.model(next_states).max(dim=1)[0] * (1 - dones)
                loss = torch.nn.MSELoss()(self.model(states).gather(1, actions), y)

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                state = next_state
                if done:
                    break
