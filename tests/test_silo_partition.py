"""Tests for silo partition logic."""

import pytest
from fedstable.data.silo_partition import partition_data_by_config


def test_partition_by_stablecoin(small_depeg_df):
    config = {
        "partition_strategy": "by_stablecoin",
        "silos": [
            {"id": 0, "name": "usdc", "filter": {"stablecoin": "USDC"}},
            {"id": 1, "name": "usdt", "filter": {"stablecoin": "USDT"}},
            {"id": 2, "name": "dai", "filter": {"stablecoin": "DAI"}},
        ],
    }
    parts = partition_data_by_config(small_depeg_df, config)
    assert len(parts) == 3
    total = sum(len(p) for p in parts.values())
    assert total == len(small_depeg_df)
    for sid, df in parts.items():
        assert df["stablecoin"].nunique() == 1


def test_partition_random(small_depeg_df):
    config = {
        "partition_strategy": "random",
        "silos": [{"id": i, "name": f"s{i}"} for i in range(3)],
    }
    parts = partition_data_by_config(small_depeg_df, config)
    assert len(parts) == 3
    total = sum(len(p) for p in parts.values())
    assert total == len(small_depeg_df)
