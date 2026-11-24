"""Feature extractor: small CNN for the danger grid + MLP for the vector."""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class BossFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, grid_features: int = 256, vec_features: int = 128):
        super().__init__(observation_space, features_dim=grid_features + vec_features)
        c, h, w = observation_space["grid"].shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, c, h, w)).shape[1]
        self.cnn_head = nn.Sequential(nn.Linear(n, grid_features), nn.ReLU())
        d = observation_space["vec"].shape[0]
        self.mlp = nn.Sequential(nn.Linear(d, vec_features), nn.ReLU(), nn.Linear(vec_features, vec_features), nn.ReLU())

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.cnn_head(self.cnn(obs["grid"])), self.mlp(obs["vec"])], dim=1)
