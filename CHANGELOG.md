# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-02-19

### Added

- **Project scaffold** — FedStable: Federated Learning for Stablecoin & DeFi Systemic Risk.
- **Specification** — Full technical spec and Q1–Q2 roadmap in README (1,100+ lines).
- **Core** — Config loading with YAML defaults, reproducibility (seed), FL utilities (get/set parameters).
- **Data** — Silo partitioning (by_stablecoin, random), schemas, loaders; graph and env data stubs for Pillar 2/3.
- **Pillar 1** — Depeg prediction: LSTM model, sequence dataset, Flower client, run_train stub; evaluation metrics (AUC-ROC, AUC-PR, Brier).
- **Pillar 2** — Contagion GNN stub, client/graph_data placeholders.
- **Pillar 3** — FedRL: StablecoinDeFiEnv, Policy stub, client placeholder.
- **Privacy** — DP gradient clipping + noise (`apply_dp_gradient`), secure aggregation interface (mask/unmask).
- **Configs** — base, pillar1/2/3, silos (3, 5, graph) YAML configs.
- **Scripts** — train_pillar1/2/3, evaluate_pillar1, download_data, preprocess (CLI entrypoints).
- **Tests** — Silo partition, Pillar 1 model shape, Pillar 3 env; pytest conftest with synthetic depeg DataFrame.
- **Documentation** — data/README, docs/PITCH placeholder; production-grade pyproject.toml and CHANGELOG.

### Technical

- Python 3.10+, Flower 1.25+, PyTorch 2.x, PyTorch Geometric; optional OmegaConf, FastAPI, Opacus.
- Designed for simulation-first, production-ready architecture (privacy, APIs, config-driven runs).

[0.1.0]: https://github.com/ansulx/Canine/releases/tag/v0.1.0
