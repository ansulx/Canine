"""Pillar 3: Policy network. Stub."""

import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list = [64, 64]):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(prev, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.mlp(x))
