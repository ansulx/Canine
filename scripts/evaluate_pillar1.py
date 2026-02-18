#!/usr/bin/env python3
"""CLI: Evaluate Pillar 1 checkpoint."""

import argparse
from pathlib import Path

# Load model and run evaluation when implemented
# from fedstable.pillar1_depeg.model import DepegPredictor
# from fedstable.evaluation.evaluate_pillar1 import evaluate_pillar1


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pillar 1 model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model.pt")
    parser.add_argument("--config", type=str, default="configs/pillar1.yaml")
    args = parser.parse_args()
    print("Evaluation stub. Implement after full train pipeline.")


if __name__ == "__main__":
    main()
