import torch


class DQNBase:
    def __init__(self, env, model):
        self.model = model
        self.env = env

    def get_q_values(self, state):
        return self.model(state.float())

    def get_action(self, state):
        return torch.argmax(self.get_q_values(state)).item()
