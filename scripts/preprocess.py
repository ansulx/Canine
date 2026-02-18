#!/usr/bin/env python3
"""Preprocess raw data and partition into silos. Stub: implement in Q1.2."""

import argparse
from pathlib import Path

from fedstable.core.config import load_config
from fedstable.core.random import set_seed


def main():
    parser = argparse.ArgumentParser(description="Preprocess and partition data")
    parser.add_argument("--config", type=str, default="configs/pillar1.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    processed_dir = Path(config.get("processed_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    print("preprocess stub. Implement in Q1.2.")


if __name__ == "__main__":
    main()
