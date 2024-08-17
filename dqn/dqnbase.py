import torch


class DQNBase:
    def __init__(self, env, model):
        self.model = model
        self.env = env

    def get_action(self, state):
        return torch.argmax(self.model(state.float())).item()
