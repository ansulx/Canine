"""Pillar 2: GNN for contagion / systemic importance. Stub for Q2."""

import torch
import torch.nn as nn


class ContagionGNN(nn.Module):
    """Graph neural network for node-level contagion score. Stub."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.linear(x).squeeze(-1)
