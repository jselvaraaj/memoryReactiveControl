import torch


class DQNTrainer:
    def __init__(self, env, model):
        self.env = env
        self.model = model

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        epooch = 1000
        for i in range(epooch):
            state = self.env.reset()
            done = False
            while not done:
                action = self.model(torch.tensor(state).float())
                next_state, reward, done, _ = self.env.step(action)
                loss = torch.nn.MSELoss()(self.model(torch.tensor(state).float()),
                                          reward + self.model(torch.tensor(next_state).float()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                state = next_state
