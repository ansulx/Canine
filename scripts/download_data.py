#!/usr/bin/env python3
"""Download raw data (CoinGecko, DeFi Llama, etc.). Stub: implement in Q1.2."""

import argparse
from pathlib import Path

from fedstable.core.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Download raw data for FedStable")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--output-dir", type=str, default=None, help="Override data_root/raw")
    args = parser.parse_args()
    config = load_config(args.config)
    raw_dir = Path(args.output_dir or config.get("raw_dir", "data/raw"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("download_data stub. Implement fetch from CoinGecko / DeFi Llama in Q1.2.")


if __name__ == "__main__":
    main()
