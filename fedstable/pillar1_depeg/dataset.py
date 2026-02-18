"""Silo-aware dataset for Pillar 1 depeg task."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class DepegDataset(Dataset):
    """Sequence dataset for depeg prediction. Loads one silo's data."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sequence_length: int = 24,
        horizon_idx: int = 1,
    ):
        """
        features: (T, F) time steps x feature dim
        labels: (T,) or (T, H) for multi-horizon; we use labels[:, horizon_idx]
        """
        self.features = features
        self.labels = labels if labels.ndim == 1 else labels[:, horizon_idx]
        self.seq_len = sequence_length
        self.horizon_idx = horizon_idx
        # Valid indices: we need seq_len + 1 past the last label we use
        self.valid_len = len(self.labels) - sequence_length
        if self.valid_len <= 0:
            raise ValueError("Sequence length too long for data")

    def __len__(self) -> int:
        return max(0, self.valid_len)

    def __getitem__(self, idx: int):
        end = idx + self.seq_len
        x = self.features[idx:end]
        y = self.labels[end - 1]
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.float32)
