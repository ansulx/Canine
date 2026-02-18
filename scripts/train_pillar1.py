#!/usr/bin/env python3
"""CLI: Train Pillar 1 (federated depeg prediction)."""

import argparse
import json
from pathlib import Path

from fedstable.pillar1_depeg.run_train import run_train
from fedstable.core.config import load_config


def main():
    parser = argparse.ArgumentParser(description="FedStable Pillar 1 training")
    parser.add_argument("--config", type=str, default="configs/pillar1.yaml", help="Config YAML path")
    parser.add_argument("overrides", nargs="*", help="Key=value overrides, e.g. training.rounds=10")
    args = parser.parse_args()
    overrides = {}
    for s in args.overrides:
        if "=" in s:
            k, v = s.split("=", 1)
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            overrides[k] = v
    config = load_config(args.config, overrides)
    artifacts_dir = Path(config.get("artifacts_dir", "artifacts")) / "pillar1"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics = run_train(args.config, overrides)
    out_path = artifacts_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved to", out_path)


if __name__ == "__main__":
    main()
