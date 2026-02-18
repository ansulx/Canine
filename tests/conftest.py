"""Pytest fixtures: small synthetic data."""

import numpy as np
import pytest


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def small_depeg_df(seed):
    """Tiny DataFrame with depeg-like columns for tests."""
    import pandas as pd
    np.random.seed(seed)
    n = 100
    return pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
        "client_id": np.random.randint(0, 3, size=n),
        "stablecoin": np.random.choice(["USDC", "USDT", "DAI"], size=n),
        "price": 1.0 + np.random.randn(n) * 0.01,
        "peg_deviation_pct": np.random.randn(n) * 0.5,
        "volume_24h": np.random.uniform(1e6, 1e8, size=n),
        "volatility_24h": np.random.uniform(0, 0.02, size=n),
        "supply": np.random.uniform(1e9, 5e9, size=n),
        "reserve_proxy": np.random.uniform(0.8, 1.0, size=n),
        "label_depeg_1h": np.random.randint(0, 2, size=n),
        "label_depeg_6h": np.random.randint(0, 2, size=n),
        "label_depeg_24h": np.random.randint(0, 2, size=n),
    })
