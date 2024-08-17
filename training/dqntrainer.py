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

        self.state_dim = self.vectorize_state(self.env.state).shape[0]

    def get_action(self, state):
        if torch.rand(1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return torch.argmax(self.model(state.float())).item()

    def get_batch(self, batch_size):
        batch = self.replayBuffer.sample(batch_size)
        return torch.stack(batch).float()

    def vectorize_sars(self, sars_with_done_flag):
        (state, action, reward, next_state, done) = sars_with_done_flag
        return torch.cat([state[None, :],
                          torch.tensor(action)[None, None],
                          torch.tensor(reward)[None, None],
                          next_state[None, :],
                          torch.tensor(1 if done else 0)[None, None]], dim=-1)[0]
        # removing the extra dimension created in the front to concat

    def devectorize_sars_from_batch(self, batch):

        state = batch[:, :self.state_dim]
        action = batch[:, self.state_dim]
        reward = batch[:, self.state_dim + 1]
        next_state = batch[:, self.state_dim + 2:-1]
        done = batch[:, -1]

        return state, action, reward, next_state, done

    def vectorize_state(self, state_dict):
        return torch.cat([torch.flatten(torch.tensor(s)) for s in state_dict.values()])

    def train(self, episode_length=1000, num_episodes=1000):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        for i in range(num_episodes):
            print(f'Episode {i + 1}')
            self.env.reset()
            state = self.vectorize_state(self.env.state)
            done = False
            for _ in range(episode_length):

                # Execute policy and collect data to add to replay buffer
                action = self.get_action(state)
                next_obs, reward, done, _ = self.env.step(action)
                next_state = self.vectorize_state(self.env.state)
                self.replayBuffer.add(self.vectorize_sars((state, action, reward, next_state, done)))

                # Sample from replay buffer and calculate loss
                batch = self.get_batch(self.batch_size)
                states, actions, rewards, next_states, dones = self.devectorize_sars_from_batch(batch)
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
            print(f'Episode {i + 1} - Loss: {loss.item()}')
            print(f'Episode {i + 1} - Replay buffer size: {len(self.replayBuffer.buffer)}')
