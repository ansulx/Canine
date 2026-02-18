"""Pillar 1: Depeg prediction model (LSTM or Transformer)."""

import torch
import torch.nn as nn
from typing import Optional


class DepegPredictor(nn.Module):
    """Temporal model for stablecoin depeg probability (multi-horizon)."""

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_horizons: int = 3,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_horizons = num_horizons
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, num_horizons)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        out: (batch, num_horizons) logits
        """
        _, (h_n, _) = self.lstm(x)
        out = self.fc(self.drop(h_n[-1]))
        return out

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities (sigmoid)."""
        return torch.sigmoid(self.forward(x))
