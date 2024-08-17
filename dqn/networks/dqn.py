import torch
import torch.nn as nn


class DQN(nn.Module):

    # The network gives q values for each action as output
    def __init__(self, input_dim, output_dim, hidden_layers=None, device=None):
        super(DQN, self).__init__()
        if hidden_layers is None:
            hidden_layers = [128, 128]

        if device is None:
            device = torch.device

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = [nn.Linear(input_dim, hidden_layers[0]), nn.ReLU()]

        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_layers[-1], output_dim))

        self.net = nn.Sequential(*layers).to(device)
        self.device = device

    def forward(self, state):
        state = state.to(self.device)
        return self.net(state)
