"""Common FL helpers: model copy, state dict to/from numpy."""

from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Return list of numpy arrays of model parameters (for Flower NumPyClient)."""
    return [p.detach().cpu().numpy() for p in model.parameters()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Set model parameters from list of numpy arrays."""
    for p, arr in zip(model.parameters(), parameters):
        p.data = torch.from_numpy(arr).to(p.device)


def state_dict_to_numpy(state_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """Convert state dict to numpy (for serialization)."""
    return {k: v.cpu().numpy() for k, v in state_dict.items()}


def numpy_to_state_dict(state_dict: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    """Convert numpy state dict back to tensors."""
    return {k: torch.from_numpy(v).to(device) for k, v in state_dict.items()}
