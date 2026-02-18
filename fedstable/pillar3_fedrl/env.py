"""Pillar 3: Simple MDP environment for stablecoin intervention. Stub."""

import numpy as np
from typing import Any, Dict, Tuple


class StablecoinDeFiEnv:
    """Discrete state/action env: state = [peg_deviation, volatility], action = {no-op, circuit_breaker}."""

    def __init__(self, state_dim: int = 4, action_dim: int = 2, max_steps: int = 100, seed: int = 42):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self._step_count = 0
        self._state = None

    def reset(self) -> np.ndarray:
        self._state = self.rng.uniform(-0.5, 0.5, size=self.state_dim).astype(np.float32)
        self._step_count = 0
        return self._state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self._step_count += 1
        # Simple dynamics: state drifts slightly; reward = -|deviation|
        self._state = self._state + self.rng.uniform(-0.05, 0.05, size=self.state_dim).astype(np.float32)
        reward = -np.abs(self._state[0])
        if action == 1:
            reward -= 0.1
        done = self._step_count >= self.max_steps
        return self._state.copy(), float(reward), done, {}
