import torch
import torch.nn as nn

from utils.utils import get_dtype


class DQN(nn.Module):

    # The network gives q values for each action as output
    def __init__(self, input_dim, output_dim, cfg, device=None):
        super(DQN, self).__init__()
        hidden_layers = cfg.policy_net.hidden_layers
        self.dtype = get_dtype(cfg.dtype)
        if device is None:
            device = torch.device

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = [nn.Linear(input_dim, hidden_layers[0]), nn.ReLU()]

        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_layers[-1], output_dim))

        self.net = nn.Sequential(*layers).to(device).to(self.dtype)
        self.device = device

    def forward(self, state):
        return self.net(state)
