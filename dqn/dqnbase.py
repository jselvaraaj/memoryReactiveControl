import torch


class DQNBase:
    def __init__(self, env, model, device):
        self.model = model
        self.env = env
        self.device = device

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        return self.model(state.to(device=self.device))

    def get_action(self, state: torch.Tensor) -> int:
        return torch.argmax(self.get_q_values(state)).item()
