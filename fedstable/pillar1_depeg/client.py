"""Flower client for Pillar 1 federated depeg training."""

from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader

from fedstable.core.fl_utils import get_parameters, set_parameters
from fedstable.pillar1_depeg.model import DepegPredictor


class DepegFlowerClient(fl.client.NumPyClient):
    """Flower client that trains DepegPredictor on local silo data."""

    def __init__(
        self,
        model: DepegPredictor,
        train_loader: DataLoader,
        device: torch.device,
        local_epochs: int = 2,
        learning_rate: float = 0.001,
    ):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return get_parameters(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        set_parameters(self.model, parameters)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        self.model.train()
        self.model.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()
        num_samples = 0
        for _ in range(self.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                logits = self.model(x)
                loss = criterion(logits[:, config.get("horizon_idx", 1)], y)
                loss.backward()
                opt.step()
                num_samples += x.size(0)
        return self.get_parameters(config), num_samples, {"loss": float(loss.item())}

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        self.model.eval()
        self.model.to(self.device)
        criterion = torch.nn.BCEWithLogitsLoss()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = criterion(logits[:, config.get("horizon_idx", 1)], y)
                total_loss += loss.item() * x.size(0)
                n += x.size(0)
        return total_loss / max(n, 1), n, {"loss": total_loss / max(n, 1)}
