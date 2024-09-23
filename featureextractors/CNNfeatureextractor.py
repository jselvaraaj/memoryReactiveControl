import torch
from torch import nn, Tensor
import gymnasium as gym


class CNNFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim: int, cnn_output_dim: int, sample_observations: Tensor, config=None):
        super().__init__()
        if config is None:
            self.cnn = nn.Sequential(
                nn.Conv2d(embedding_dim, 32, kernel_size=3, padding=0),
                nn.ReLU(),
                # nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                # nn.ReLU(),
                # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                # nn.ReLU(),
                nn.Flatten(),
            )
        else:
            layers = []
            for layer_cfg in config.conv_layers:
                layers.append(nn.Conv2d(layer_cfg.in_channels,
                                        layer_cfg.out_channels,
                                        kernel_size=layer_cfg.kernel_size,
                                        stride=layer_cfg.stride,
                                        padding=layer_cfg.padding))
                layers.append(nn.ReLU())
            layers.append(nn.Flatten())
            self.cnn = nn.Sequential(*layers)

        with torch.no_grad():
            sample_observations = sample_observations.to(torch.float32)
            n_flatten = self.cnn(sample_observations).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, cnn_output_dim), nn.ReLU())

    def forward(self, x):
        x = x.to(torch.float32)
        return self.linear(self.cnn(x))
