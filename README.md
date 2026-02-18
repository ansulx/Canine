# FedStable: Federated Learning for Stablecoin & DeFi Systemic Risk

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-informational)](CHANGELOG.md)

**Production-grade, privacy-preserving ML for stablecoin depeg and DeFi systemic risk — no raw data sharing.**

---

## What We Do

| | |
|--|--|
| **Product** | Federated learning platform that predicts **stablecoin depeg risk** and **DeFi contagion** across institutions (exchanges, protocols, custodians) **without centralizing data**. |
| **Why it matters** | Regulators and issuers need cross-institutional risk signals; data cannot be pooled. We train one model from many silos via **gradient/weight aggregation only** — privacy and compliance by design. |
| **Three pillars** | **(1)** Depeg & reserve-stress prediction (time-series FL). **(2)** Contagion and systemic importance (federated graph learning). **(3)** Intervention policies (federated reinforcement learning). |
| **Target users** | Stablecoin issuers, exchanges, DeFi protocols, central banks / regulators. |
| **Tech** | Flower + PyTorch; optional differential privacy and secure aggregation; config-driven; reproducible. |

**This repository** is the canonical specification, codebase, and roadmap (Q1–Q2) for the FedStable system. See [CHANGELOG](CHANGELOG.md) for version history and [LICENSE](LICENSE) for terms.

---

**Specification & Development Roadmap (Q1–Q2)**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Goals, Scope, and Non-Goals](#3-goals-scope-and-non-goals)
4. [Architecture Overview](#4-architecture-overview)
5. [Pillar 1: Federated Stablecoin Depeg Prediction](#5-pillar-1-federated-stablecoin-depeg-prediction)
6. [Pillar 2: Federated Graph Learning for DeFi Contagion](#6-pillar-2-federated-graph-learning-for-defi-contagion)
7. [Pillar 3: Federated RL for Protocol Interventions](#7-pillar-3-federated-rl-for-protocol-interventions)
8. [Tech Stack and Integrations](#8-tech-stack-and-integrations)
9. [Data Pipeline and Silo Simulation](#9-data-pipeline-and-silo-simulation)
10. [Privacy and Security](#10-privacy-and-security)
11. [Reproducibility and Experimentation](#11-reproducibility-and-experimentation)
12. [API and Interface Design](#12-api-and-interface-design)
13. [Directory and File Structure](#13-directory-and-file-structure)
14. [Q1–Q2 Development Roadmap](#14-q1q2-development-roadmap)
15. [Testing Strategy](#15-testing-strategy)
16. [Deployment and Operations](#16-deployment-and-operations)
17. [References and Prior Art](#17-references-and-prior-art)
18. [Glossary](#18-glossary)

---

## 1. Project Overview

### 1.1 One-Line Description

FedStable is a federated learning system that predicts stablecoin depeg risk and DeFi systemic contagion across siloed institutions (exchanges, protocols, custodians) without raw data sharing, with an optional federated reinforcement learning module for stability interventions.

### 1.2 Motivation

- **Data silos:** Exchanges, custodians, and DeFi protocols each hold partial views of order flow, reserves, and exposures. No single party has a complete picture for depeg or contagion prediction.
- **Regulatory and commercial constraints:** Raw data cannot be centralized; privacy and confidentiality are mandatory.
- **Research gap:** Existing work on stablecoin depeg prediction and DeFi systemic risk uses centralized data. Federated learning for this domain is largely unexplored.
- **Impact:** Enables collaborative early-warning and risk monitoring for stablecoin issuers, exchanges, protocols, and regulators.

### 1.3 Three Pillars (Summary)

| Pillar | Objective | Output |
|--------|-----------|--------|
| **P1** | Federated stablecoin depeg & reserve-stress prediction | Shared or personalized model; depeg probability / stress score over 1h–24h horizons |
| **P2** | Federated graph learning for DeFi contagion | Contagion risk and systemic importance scores without sharing full graph |
| **P3** | Federated RL for protocol interventions | Policy for when to trigger circuit breakers or parameter changes to reduce depeg/contagion |

---

## 2. Problem Statement

### 2.1 Technical Problem

- **Input:** Multiple institutions (clients), each with local time-series and/or graph data related to stablecoin flows, reserves, and DeFi exposures. Data is non-IID (different assets, geographies, user bases) and may be temporal and graph-structured.
- **Constraints:** No raw data leaves any client; only model updates (gradients or weights) or aggregated statistics may be exchanged. Compliance with privacy requirements (e.g., differential privacy, secure aggregation) may be required.
- **Output:** (1) A predictive model for stablecoin depeg / reserve stress usable by all or by each client with personalization; (2) Contagion and systemic-importance estimates; (3) Optionally, a policy for stability interventions.

### 2.2 Assumptions

- Clients participate voluntarily and do not deliberately poison the federation (or we consider Byzantine-robust aggregation as an extension).
- Sufficient public or synthetic data exists to simulate silos and validate methods before real multi-party deployment.
- Communication and compute budgets are finite; communication-efficient FL is desirable.
- We target initially a simulation environment (siloed splits of public data); production deployment with real institutions is a later phase.

---

## 3. Goals, Scope, and Non-Goals

### 3.1 Goals

- Implement and evaluate **Pillar 1** (federated depeg prediction) with Flower + PyTorch on public/synthetic siloed data.
- Implement and evaluate **Pillar 2** (federated graph learning for contagion) with a clear baseline comparison to centralized graph models.
- Implement and evaluate **Pillar 3** (FedRL for interventions) in a simulated environment with heterogeneous clients.
- Provide a **reproducible** codebase: configs, seeds, data splits, and evaluation scripts.
- Document APIs, directory layout, and integration points for external data and future productionization.

### 3.2 Scope (In Scope for Q1–Q2)

- All three pillars in **simulation mode** (silo simulation from public/synthetic data).
- Flower as the primary FL framework; PyTorch for models.
- Differential privacy (DP) and optional secure aggregation integration points.
- Graph learning via PyTorch Geometric or equivalent; time-series via PyTorch.
- CLI and config-driven experiments; optional REST API for inference.
- Comprehensive specification (this document) and a tech-focused pitch (deck or doc) for Q2.

### 3.3 Non-Goals (Explicitly Out of Scope for Q1–Q2)

- Production deployment at real institutions (no live exchange/protocol integrations).
- Blockchain-based aggregation (e.g., FLock-style) unless added as an optional experiment.
- Full formal security proofs (we rely on standard DP and aggregation; proofs are future work).
- Marketing or business development content in this specification; focus is technical.

---

## 4. Architecture Overview

### 4.1 System Context

```
                    +------------------+
                    |  FedStable       |
                    |  Orchestration   |
                    |  (Flower Server  |
                    |   + our logic)   |
                    +--------+---------+
                             |
         +-------------------+-------------------+
         |                   |                   |
    +----v----+         +----v----+         +----v----+
    | Client 1|         | Client 2|         | Client N|
    | Exchange|         | Protocol|         |Custodian |
    | (silo 1)|         | (silo 2)|         | (silo N)|
    +----+----+         +----+----+         +----+----+
         |                   |                   |
         |  local data       |  local data       |  local data
         |  (no raw export)  |  (no raw export)  |  (no raw export)
         v                   v                   v
    [ gradients / weights / encrypted aggregates only ]
```

- **Server:** Coordinates rounds, aggregates updates, distributes global (or personalized) model. May hold evaluation data that is not used for training (e.g., public test set).
- **Clients:** Train locally on their silo; send only model updates (and optionally DP noise or encrypted shares). Never send raw time-series or graph data.

### 4.2 Component Diagram

```
fedstable/
├── core/                 # FL orchestration, strategy, aggregation
├── models/               # PyTorch models (depeg, graph, policy)
├── data/                 # Data loaders, silo partitioning, preprocessing
├── pillar1_depeg/        # Pillar 1 pipeline and configs
├── pillar2_contagion/    # Pillar 2 pipeline and configs
├── pillar3_fedrl/        # Pillar 3 pipeline and configs
├── privacy/              # DP, secure aggregation interfaces
├── evaluation/           # Metrics, plots, reporting
├── api/                  # Optional REST/CLI for inference and training
├── configs/              # YAML/JSON experiment configs
├── scripts/              # Entrypoints (train, evaluate, simulate_silos)
└── tests/                # Unit and integration tests
```

### 4.3 Data Flow (High Level)

1. **Offline:** Public/synthetic data → preprocessing → partitioned into silos (by “institution” or by asset/venue) → saved in a standard format (e.g., Parquet + metadata).
2. **Training:** Each client loads its silo; server runs rounds; clients train locally and submit updates; server aggregates and broadcasts; repeat until convergence or max rounds.
3. **Evaluation:** Global (or per-client) model is evaluated on held-out test data (central test set or per-silo test sets). Metrics: AUC, calibration, MAE for stress scores, etc.
4. **Inference (optional):** A served model accepts feature vectors (from any client) and returns depeg probability or stress score; no raw data leaves the client in production.

---

## 5. Pillar 1: Federated Stablecoin Depeg Prediction

### 5.1 Objective

Train a model that predicts stablecoin depeg probability (or reserve stress score) over horizons 1h, 6h, 24h, using only federated updates from multiple silos. No raw order flow, reserves, or redemptions are shared.

### 5.2 Model Architecture (Technical)

- **Inputs (per time step or window):** Features derived from (simulated) local data: price deviation from peg, volume, volatility, redemption-like signals, reserve proxy, and optionally contagion scores from Pillar 2.
- **Architecture:** Temporal model (e.g., LSTM, Transformer, or 1D CNN + RNN) that consumes a fixed-length history (e.g., 24–168 steps) and outputs either:
  - **Classification:** Probability of depeg in next 1h / 6h / 24h (binary or multi-horizon).
  - **Regression:** Reserve stress score or peg deviation magnitude.
- **Output:** Single head or multi-head for multi-horizon. Loss: BCE for classification; MSE or Huber for regression. Optional auxiliary losses for calibration.

### 5.3 Federated Learning Strategy

- **Algorithm:** FedAvg or FedProx (with optional local steps and proximal term). Future: FedOpt (e.g., server-side Adam), SCAFFOLD, or personalized variants (e.g., per-client heads).
- **Client selection:** Uniform random or weighted by dataset size; optional straggler handling (timeout, partial updates).
- **Aggregation:** Weighted average by number of local samples (or by loss weights). Optional: differential privacy (DP-SGD on client, then clip-and-noisy aggregate on server) or secure aggregation ( cryptographic protocol so server only sees sum of updates).

### 5.4 Data (Pillar 1)

- **Sources (simulation):** Public on-chain data (e.g., stablecoin supply, exchange flows from public APIs), public order book snapshots or OHLCV, and synthetic redemptions/reserve signals. For true silo simulation, partition by “institution” (e.g., by exchange, or by stablecoin, or by synthetic split).
- **Labels:** Depeg events from historical data (e.g., USDC March 2023, UST collapse). Define depeg as deviation beyond a threshold (e.g., ±1% or ±2%) for a minimum duration.
- **Train/val/test:** Split by time (e.g., last 20% test, previous 10% val). Ensure no leakage across silos (each silo’s train/val/test are disjoint in time if needed).

### 5.5 Evaluation (Pillar 1)

- **Metrics:** AUC-ROC, AUC-PR, Brier score, calibration curve; for regression, MAE, RMSE, correlation with realized stress.
- **Baselines:** (1) Local-only (each client trains on its silo only, no FL). (2) Centralized (pool all data, train one model—upper bound in data regime). (3) FedAvg vs FedProx vs personalized.
- **Reporting:** Per-silo and pooled test performance; communication cost (number of rounds × bytes per round); wall-clock time.

### 5.6 File Paths (Pillar 1)

- `fedstable/pillar1_depeg/model.py` — Model definition (PyTorch).
- `fedstable/pillar1_depeg/client.py` — Flower client logic (train one round, return weights).
- `fedstable/pillar1_depeg/server.py` — Server strategy and aggregation (or use Flower built-in + custom strategy).
- `fedstable/pillar1_depeg/dataset.py` — Silo-aware dataset loading for depeg task.
- `fedstable/pillar1_depeg/config.yaml` — Default hyperparameters (learning rate, local epochs, batch size, etc.).
- `fedstable/data/silo_partition.py` — Partitioning logic (used by all pillars where applicable).

---

## 6. Pillar 2: Federated Graph Learning for DeFi Contagion

### 6.1 Objective

Learn a model that estimates contagion risk and/or systemic importance of nodes (protocols, pools, stablecoins) in a DeFi graph, without any party sharing its full subgraph. Each client holds a subgraph (e.g., one protocol’s view of exposures); the federation learns a global representation.

### 6.2 Graph Model (Technical)

- **Graph structure:** Nodes = protocols, pools, or aggregated “institutions”; edges = exposures, shared collateral, or flow volumes. Time-varying edge weights optional (temporal graph).
- **Model:** Graph Neural Network (GNN), e.g., GCN, GraphSAGE, or temporal GNN (e.g., TGNN, or GNN over snapshots). Input: node features (e.g., TVL proxy, volume); output: node-level contagion risk score or systemic importance score.
- **Federated setting:** Each client has a subgraph (subset of nodes and edges). Overlapping nodes across clients are allowed; we do not require disjoint node sets. Aggregation is over model parameters, not over raw graph structure.

### 6.3 Silo Simulation (Pillar 2)

- **Partitioning:** By protocol category, or by “owner” (e.g., client 1 sees only Aave-related nodes and edges, client 2 sees only Uniswap-related). Alternatively, random partition of edges so each client sees a subset of the global graph.
- **Challenge:** Non-IID structure (different clients have different degree distributions and local topology). Document how we partition and what overlap exists.

### 6.4 Aggregation and Training

- **Strategy:** FedAvg over GNN parameters. Gradients are computed on local subgraphs; server aggregates. Consider gradient clipping for stability.
- **Optional:** Cross-client node alignment (if node IDs are shared but features are local, we need a consistent ID space; if not, we rely on structure and feature aggregation only).

### 6.5 Evaluation (Pillar 2)

- **Metrics:** Correlation with centralized “oracle” contagion score (computed on full graph when available for evaluation only); node-level ranking metrics (e.g., NDCG for “top-k systemic nodes”); link prediction or stress propagation accuracy if we have ground truth.
- **Baselines:** Centralized GNN on full graph; local-only GNN per client; our federated GNN.

### 6.6 File Paths (Pillar 2)

- `fedstable/pillar2_contagion/model.py` — GNN model (PyTorch Geometric or custom).
- `fedstable/pillar2_contagion/client.py` — Flower client for graph training.
- `fedstable/pillar2_contagion/graph_data.py` — Load and partition graph data per silo.
- `fedstable/pillar2_contagion/config.yaml` — GNN and FL hyperparameters.
- `fedstable/data/graph_sources.py` — Build graph from public DeFi datasets (e.g., DeFi Llama, or synthetic).

---

## 7. Pillar 3: Federated RL for Protocol Interventions

### 7.1 Objective

Learn a shared (or personalized) policy that, given local state (e.g., collateral ratios, flows, oracle prices), outputs discrete or continuous interventions (e.g., circuit breaker on/off, collateral factor adjustment) to reduce depeg probability or contagion in simulation.

### 7.2 Environment (Technical)

- **MDP:** States = (aggregate collateral ratio, stablecoin supply, price deviation, volatility proxy, optional contagion score from Pillar 2). Actions = discrete (e.g., no-op, trigger circuit breaker, reduce max LTV) or continuous (e.g., target collateral ratio). Rewards = negative of depeg severity or negative of contagion spread; optional penalty for excessive intervention.
- **Simulator:** Use a simplified model of stablecoin and DeFi (e.g., agent-based or equation-based) parameterized by public data. Multiple environments = different protocol parameters or different random seeds (heterogeneous clients).

### 7.3 Policy and Algorithm

- **Policy:** Neural network (MLP or small Transformer) mapping state to action distribution (and value if using actor-critic). Input may include shared but privacy-safe statistics (e.g., federated mean of a stress index) in addition to local state.
- **Algorithm:** Federated policy gradient or FedSVRPG-style (variance-reduced FedRL). Each client runs local trajectories; computes policy gradient; server aggregates gradients and updates global policy. Optional: personalized layers per client.

### 7.4 Evaluation (Pillar 3)

- **Metrics:** Average return (sum of rewards per episode); depeg frequency and severity in evaluation episodes; intervention frequency (sparsity). Compare FedRL vs local-only RL vs no intervention.

### 7.5 File Paths (Pillar 3)

- `fedstable/pillar3_fedrl/env.py` — Simulation environment (state, action, reward, step).
- `fedstable/pillar3_fedrl/policy.py` — Policy network and value head.
- `fedstable/pillar3_fedrl/client.py` — Flower client (collect trajectories, compute gradient, send update).
- `fedstable/pillar3_fedrl/config.yaml` — Env and FedRL hyperparameters.
- `fedstable/data/env_data.py` — Parameters for simulator derived from public data.

---

## 8. Tech Stack and Integrations

### 8.1 Core

- **Language:** Python 3.10+.
- **FL framework:** [Flower](https://flower.ai/) (flwr). Use `flwr.server` and `flwr.client`; implement `Client` and `Strategy` as needed. Version: align with Flower 1.x (e.g., 1.25+).
- **Deep learning:** PyTorch 2.x. Use `torch`, `torch.nn`, and standard optimizers. Device: CPU and CUDA where available.
- **Experiment management:** Configs in YAML (or JSON); optional Hydra or OmegaConf for overrides. No mandatory experiment tracker for Q1–Q2, but structure logs and artifacts so that MLflow or W&B can be added later.

### 8.2 Data and Graph

- **Data handling:** pandas, numpy; optional PyArrow/Parquet for large tables. On-chain data: use public APIs (e.g., CoinGecko, DeFi Llama, or on-chain RPC) via a thin adapter layer so sources can be swapped.
- **Graph:** PyTorch Geometric (torch_geometric) for GNNs and graph data structures. Alternative: DGL if we need specific ops. Version: compatible with current PyTorch.
- **Time-series:** Custom dataloaders with `torch.utils.data.Dataset`; sliding windows for sequences. No mandatory extra library; optional: tsai or sktime if we add more advanced temporal models later.

### 8.3 Privacy

- **Differential privacy:** PyTorch Opacus or custom DP-SGD (gradient clipping + noise). Interface: optional wrapper that adds clipping and noise to client updates before send. Target: (epsilon, delta)-DP per round or over training.
- **Secure aggregation:** Interface only in Q1–Q2 (e.g., placeholder that “would call” a secure aggregation protocol). No mandatory dependency on a specific crypto library; document the API so that integration (e.g., with Flower’s secagg or custom MPC) can be added later.

### 8.4 Evaluation and Logging

- **Metrics:** sklearn (AUC, etc.), custom functions for calibration and stress metrics. Plots: matplotlib or seaborn; save figures under `artifacts/` or `results/`.
- **Logging:** Python `logging` with levels; optional structlog. No raw data in logs; only aggregates and metrics.
- **Reproducibility:** Set seeds for numpy, torch, and Python random in a single `fedstable/core/random.py` or in each script entrypoint.

### 8.5 API and CLI

- **CLI:** Entrypoints via `scripts/train_pillar1.py`, `scripts/train_pillar2.py`, `scripts/train_pillar3.py`, and `scripts/evaluate_*.py`. Arguments: config path, overrides (e.g., `--config configs/pillar1.yaml data.silo_config=configs/silos_3.yaml`).
- **REST API (optional):** FastAPI or Flask app under `fedstable/api/` that: (1) loads a trained model, (2) accepts feature vector(s), (3) returns depeg probability or stress score. Auth and rate limiting are deployment concerns; not specified in detail here.

### 8.6 Integrations Summary Table

| Area | Technology | Purpose |
|------|------------|---------|
| FL | Flower (flwr) | Server-client orchestration, strategies |
| DL | PyTorch | Models, optimizers, autograd |
| Graph | PyTorch Geometric | GNNs, graph loaders |
| DP | Opacus or custom | Client-side DP-SGD |
| Config | YAML + OmegaConf/Hydra (optional) | Experiment configs |
| Data I/O | pandas, PyArrow, requests | Tables, Parquet, API calls |
| Eval | sklearn, custom | AUC, calibration, stress metrics |
| API | FastAPI (optional) | Inference endpoint |

---

## 9. Data Pipeline and Silo Simulation

### 9.1 Data Sources (Planned)

- **Stablecoin prices and supply:** Public APIs (e.g., CoinGecko, CoinMarketCap) or on-chain (Ethereum, etc.). Store: timestamp, symbol, price, supply, volume.
- **DeFi TVL and flows:** DeFi Llama or similar. Protocols, pools, TVL over time.
- **Order book / OHLCV:** Public exchange APIs or archived data (e.g., CryptoDataDownload). Used for volatility and volume features.
- **Historical depeg events:** Curated list (date, stablecoin, severity, duration) from public reports. Used as labels for Pillar 1.
- **Graph construction (Pillar 2):** Nodes = protocols or pools; edges = collateral links, shared assets, or flow volumes. Built from DeFi Llama and/or synthetic edges for controlled experiments.

### 9.2 Preprocessing

- **Pillar 1:** Resample to fixed frequency (e.g., 1h). Compute features: return, volatility (rolling), volume change, deviation from peg, optional lagged contagion index. Align with label horizon (e.g., label = 1 if depeg in next 6h).
- **Pillar 2:** Build adjacency and node features; normalize; optionally create temporal snapshots.
- **Pillar 3:** State and reward from same data sources; simulator uses parameterized dynamics.

### 9.3 Silo Partitioning

- **By institution (simulated):** Assign each sample or node to a “client” by a rule: e.g., by exchange (if we have exchange ID), by stablecoin, or by random partition. Ensure no sample is in two clients.
- **By time (optional):** For stress tests, partition by time so that some clients have only “normal” and others “crisis” periods to test non-IID robustness.
- **Config:** Silo partition defined in a config file (e.g., `silos: [{name: exchange_a, data_path: ...}, ...]` or `partition_strategy: by_stablecoin`). Code in `fedstable/data/silo_partition.py` reads config and produces per-client datasets.

### 9.4 Data Formats

- **Tabular (Pillar 1):** Parquet or CSV with columns: `timestamp`, `client_id`, feature columns, `label_depeg_1h`, `label_depeg_6h`, `label_depeg_24h` (or single label column). Schema documented in `fedstable/data/schemas.py` or in this README.
- **Graph (Pillar 2):** Save per-silo `Data` objects (PyG) or edge index + node features in NumPy/Parquet; loader assembles them. Global graph (for centralized baseline) stored separately for evaluation.
- **RL (Pillar 3):** Simulator generates states on the fly; no mandatory file format. Parameters (e.g., mean reversion, shock size) in config or small JSON.

### 9.5 Caching and Versioning

- Raw and processed data under `data/raw/` and `data/processed/` (or paths in config). Document expected directory layout. Optional: DVC or symlinks for large assets; version data with a `data_version` key in config for reproducibility.

---

## 10. Privacy and Security

### 10.1 Threat Model

- **Server:** Honest-but-curious (aggregates correctly but may try to infer client data from updates). No raw data is sent; gradient or weight updates may leak information—hence optional DP.
- **Clients:** Do not collude in Q1–Q2; we do not assume Byzantine clients by default. Optional: robust aggregation (e.g., median, Krum) as future work.
- **Network:** Channel between client and server is authenticated and encrypted in production (TLS); out of scope for simulation.

### 10.2 Differential Privacy

- **Client-level DP:** Each client runs DP-SGD (clip gradients, add Gaussian noise) and sends noisy update. Server aggregates as usual. Parameters: clip norm C, noise scale sigma, target (epsilon, delta). Document in config and in evaluation (privacy-utility trade-off curves).
- **Optional server-side DP:** Noisy aggregation at server (e.g., add noise to aggregated gradient). Interface in `fedstable/privacy/dp.py`.

### 10.3 Secure Aggregation (Interface Only)

- **API:** `fedstable/privacy/secure_agg.py` defines an interface: `mask_gradient(grad, client_id, round_id) -> masked_grad` and `unmask_aggregate(masked_grads) -> aggregate`. Implementation can be a no-op (return as-is) or later wired to a real secure aggregation protocol. Document so that integration is straightforward.

### 10.4 Data Minimization

- Logs and saved artifacts must not contain raw user or transaction-level data. Only aggregates, metrics, and model weights (and optionally sanitized feature distributions for debugging) are allowed.

---

## 11. Reproducibility and Experimentation

### 11.1 Random Seeds

- Single entry point: `fedstable/core/random.py` with `set_seed(seed)` that sets `numpy.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, and `random.seed`. Every train and eval script calls this with a configurable seed (default in config).

### 11.2 Config Structure

- **Global:** `configs/base.yaml` — seed, device, log dir, data root.
- **Per-pillar:** `configs/pillar1.yaml`, `configs/pillar2.yaml`, `configs/pillar3.yaml` — model, FL (rounds, clients per round, local epochs), data paths, silo config path.
- **Override from CLI:** e.g., `python scripts/train_pillar1.py --config configs/pillar1.yaml training.rounds=100 data.silo_config=configs/silos_5.yaml`.

### 11.3 Artifacts

- **Output dir:** `artifacts/` or `results/` (configurable). Subdirs: `pillar1/run_<timestamp>/`, etc. Contents: config copy, checkpoint (best or last), metrics JSON, plots. Naming convention: `metrics.json`, `model.pt`, `config.yaml`.

### 11.4 Versioning

- Pin dependencies in `requirements.txt` or `pyproject.toml` (Python 3.10+, torch, flwr, torch_geometric, pandas, etc.). Document in README. Optional: conda `environment.yml` for full reproducibility.

---

## 12. API and Interface Design

### 12.1 Client API (Flower)

- Each pillar implements a Flower `Client` that:
  - Receives global model (or current round config) from server.
  - Loads local dataset for that client ID.
  - Trains for configured local steps/epochs.
  - Returns `ClientRes` with updated weights (or gradients, depending on strategy). Optional: local metrics (loss, accuracy) in res for server-side logging.
- Same client class can be used in simulation by launching N processes or threads, each with a different `client_id` and data partition.

### 12.2 Server API (Flower)

- Use Flower’s `ServerConfig`, `Strategy`, and `start_server`. Custom strategy if we need FedProx, personalization, or DP aggregation; otherwise `FedAvg`. Strategy receives client results and returns aggregated model. We implement `evaluate` callback for central evaluation (optional central test set).

### 12.3 Inference API (Optional REST)

- **Endpoint:** `POST /predict` — body: `{"features": [[...]], "model": "depeg"}`. Response: `{"probabilities": [...], "model_version": "..."}`. Load model from artifact path (env var or config). No auth specified here; add in deployment.

### 12.4 Internal Python API

- **Training:** `fedstable.pillar1_depeg.run_train(config) -> metrics_dict`. Same pattern for Pillar 2 and 3. Used by `scripts/train_*.py` and by tests.
- **Evaluation:** `fedstable.evaluation.evaluate_pillar1(model, dataloader, device) -> dict`. Shared evaluation helpers in `fedstable/evaluation/`.

---

## 13. Directory and File Structure

Full target layout with short descriptions. Paths are relative to repository root.

```
Canine/
├── README.md                    # This specification
├── requirements.txt             # Python dependencies (versions pinned)
├── pyproject.toml               # Optional: project metadata, build
├── .env.example                 # Example env vars (data paths, API keys if any)
│
├── configs/
│   ├── base.yaml                # Global: seed, device, paths
│   ├── pillar1.yaml             # Pillar 1 default hyperparameters
│   ├── pillar2.yaml             # Pillar 2 default hyperparameters
│   ├── pillar3.yaml             # Pillar 3 default hyperparameters
│   ├── silos_3.yaml             # Example: 3 silos for simulation
│   ├── silos_5.yaml             # Example: 5 silos
│   └── dp.yaml                  # Optional: DP parameters (C, sigma, epsilon)
│
├── data/                        # Data dir (gitignored or DVC); structure only
│   ├── raw/                     # Raw downloads (CSV, JSON from APIs)
│   │   ├── coingecko/           # Price, supply, volume
│   │   ├── defillama/           # TVL, protocols
│   │   └── depeg_events.json    # Curated depeg labels
│   ├── processed/               # Preprocessed tables and graph
│   │   ├── depeg_features.parquet
│   │   ├── graph_global.pt      # Full graph for baseline eval
│   │   └── silo_0/             # Per-silo data (Pillar 1)
│   │       ├── train.parquet
│   │       └── test.parquet
│   └── README.md                # Data schema and how to generate
│
├── fedstable/
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── random.py            # set_seed()
│   │   ├── config.py            # Load and merge configs
│   │   └── fl_utils.py          # Common FL helpers (e.g., model copy, state dict)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── temporal.py          # LSTM/Transformer for Pillar 1 (or in pillar1)
│   │   └── gnn.py               # GNN base for Pillar 2 (or in pillar2)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schemas.py           # Column names, dtypes (doc or validation)
│   │   ├── loaders.py           # Generic tabular and sequence loaders
│   │   ├── silo_partition.py    # Partition data by silo config
│   │   ├── graph_sources.py     # Build DeFi graph from raw sources
│   │   └── env_data.py          # Simulator parameters for Pillar 3
│   │
│   ├── pillar1_depeg/
│   │   ├── __init__.py
│   │   ├── model.py             # DepegPredictor (PyTorch Module)
│   │   ├── dataset.py           # Silo-aware dataset for depeg
│   │   ├── client.py            # Flower client
│   │   ├── server.py            # Strategy / server config (or use flwr default)
│   │   ├── run_train.py         # run_train(config) -> metrics
│   │   └── config.yaml          # Pillar-1-specific defaults (or in configs/)
│   │
│   ├── pillar2_contagion/
│   │   ├── __init__.py
│   │   ├── model.py             # ContagionGNN (PyTorch Geometric)
│   │   ├── graph_data.py        # Load per-silo graph
│   │   ├── client.py            # Flower client
│   │   ├── server.py            # Strategy
│   │   └── run_train.py         # run_train(config) -> metrics
│   │
│   ├── pillar3_fedrl/
│   │   ├── __init__.py
│   │   ├── env.py               # StablecoinDeFiEnv (MDP)
│   │   ├── policy.py            # Policy and value networks
│   │   ├── client.py            # Flower client (trajectories -> gradient)
│   │   ├── server.py            # Strategy
│   │   └── run_train.py         # run_train(config) -> metrics
│   │
│   ├── privacy/
│   │   ├── __init__.py
│   │   ├── dp.py                # DP-SGD wrapper (clip + noise)
│   │   └── secure_agg.py        # Secure aggregation interface (stub or real)
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # AUC, calibration, stress metrics
│   │   ├── evaluate_pillar1.py   # evaluate_pillar1(model, loader) -> dict
│   │   ├── evaluate_pillar2.py   # evaluate_pillar2(model, graph_loader) -> dict
│   │   └── evaluate_pillar3.py   # evaluate_pillar3(policy, env) -> dict
│   │
│   └── api/
│       ├── __init__.py
│       ├── app.py               # FastAPI app (optional)
│       └── inference.py         # load model, run inference
│
├── scripts/
│   ├── train_pillar1.py         # CLI: load config, run_train pillar1
│   ├── train_pillar2.py         # CLI: pillar2
│   ├── train_pillar3.py         # CLI: pillar3
│   ├── evaluate_pillar1.py       # Load checkpoint, run evaluation
│   ├── evaluate_pillar2.py
│   ├── evaluate_pillar3.py
│   ├── download_data.py         # Fetch raw data (APIs, etc.)
│   ├── preprocess.py            # Raw -> processed, silo partition
│   └── run_simulation.py        # Optional: single script that runs N clients in process
│
├── tests/
│   ├── conftest.py              # Pytest fixtures (small synthetic data)
│   ├── test_silo_partition.py   # Partition logic
│   ├── test_pillar1_model.py    # Forward pass, shape
│   ├── test_pillar2_model.py
│   ├── test_pillar3_env.py      # Env step, reward
│   ├── test_fl_integration.py   # 2 clients, 2 rounds, check aggregation
│   └── test_dp.py               # DP wrapper adds noise, reduces norm
│
├── artifacts/                   # Output dir (gitignored)
│   ├── pillar1/
│   ├── pillar2/
│   └── pillar3/
│
├── docs/                        # Optional extra docs
│   ├── SPECIFICATION.md         # Copy or link to README sections
│   └── PITCH.md                 # Tech-focused pitch (Q2)
│
└── .gitignore                   # data/raw, data/processed, artifacts, .env, __pycache__
```

---

## 14. Q1–Q2 Development Roadmap

### 14.1 Timeline Overview

- **Q1:** Foundation, Pillar 1 complete, Pillar 2 and 3 scaffolded and minimally runnable.
- **Q2:** Pillar 2 and 3 complete; evaluation and baselines; reproducibility; tech pitch and documentation.

### 14.2 Q1 Milestones (Detailed)

#### Q1.1 — Repo and core (Weeks 1–2)

- **Deliverables:**
  - Repository structure as in Section 13 (empty stubs OK for pillar2/3).
  - `requirements.txt` / `pyproject.toml` with Flower, PyTorch, torch_geometric, pandas, numpy, pyyaml. Optional: OmegaConf, FastAPI.
  - `fedstable/core/random.py`, `config.py`, `fl_utils.py` implemented.
  - `configs/base.yaml` and `configs/pillar1.yaml` with all Pillar 1 hyperparameters (model, FL, data paths).
  - `data/README.md` with schema for depeg features and labels.
- **Acceptance:** `python -c "from fedstable.core.config import load_config; print(load_config('configs/base.yaml'))"` runs; `set_seed(42)` runs without error.

#### Q1.2 — Data pipeline and silos (Weeks 2–3)

- **Deliverables:**
  - `scripts/download_data.py`: fetch stablecoin price/supply and (if possible) a public order-book or OHLCV source; save under `data/raw/`. Curated `depeg_events.json` (at least USDC March 2023, optionally more).
  - `fedstable/data/loaders.py`: load tabular data; `fedstable/data/silo_partition.py`: partition by config (e.g., by_stablecoin or random) into N silos; write `data/processed/silo_<i>/train.parquet`, `test.parquet`.
  - `scripts/preprocess.py`: orchestrate preprocessing and partitioning from config. Document feature list and label definition in `data/README.md`.
- **Acceptance:** Running `preprocess.py` with a small raw sample produces at least 2 silos with train/test; schema documented.

#### Q1.3 — Pillar 1 model and dataset (Weeks 3–4)

- **Deliverables:**
  - `fedstable/pillar1_depeg/model.py`: PyTorch model (e.g., LSTM or Transformer) with configurable input size, hidden size, horizons. Output: logits or probabilities for depeg.
  - `fedstable/pillar1_depeg/dataset.py`: Dataset that loads one silo’s Parquet, builds sequences (sliding window), returns (X, y). Integrate with `silo_partition` (path from config).
  - Unit test: forward pass with dummy tensor; dataset returns correct shapes.
- **Acceptance:** `train_pillar1.py` can run in “centralized” mode (single client, all data) for 2–3 epochs and loss decreases.

#### Q1.4 — Pillar 1 federated training (Weeks 4–6)

- **Deliverables:**
  - `fedstable/pillar1_depeg/client.py`: Flower client; gets global model, trains on local dataset (by client_id), returns updated parameters.
  - `fedstable/pillar1_depeg/server.py` or use `flwr.server.Server` with `FedAvg`. Strategy: weighted average by sample count.
  - `fedstable/pillar1_depeg/run_train.py`: load config, start server, register N clients (in-process or multi-process), run R rounds, save checkpoint and metrics.
  - `scripts/train_pillar1.py`: CLI with config path and overrides. Saves to `artifacts/pillar1/run_<timestamp>/`.
- **Acceptance:** 3 silos, 10 rounds; training completes; saved `metrics.json` contains train/val loss or accuracy per round (or central eval).

#### Q1.5 — Pillar 1 evaluation and baselines (Weeks 5–6)

- **Deliverables:**
  - `fedstable/evaluation/metrics.py`: AUC-ROC, AUC-PR, Brier score; optional calibration plot.
  - `fedstable/evaluation/evaluate_pillar1.py`: load model checkpoint, load test set(s), compute metrics. Support both pooled test and per-silo test.
  - `scripts/evaluate_pillar1.py`: CLI (checkpoint path, config). Baseline: run centralized training (single client with all data), evaluate; document in README as “centralized baseline”.
  - Optional: local-only baseline (each silo trains alone, report average test metric).
- **Acceptance:** Evaluation script runs on a saved checkpoint; metrics match expected format. README states how to reproduce centralized vs FedAvg.

#### Q1.6 — Pillar 2 and 3 scaffold (Weeks 5–6)

- **Deliverables:**
  - Pillar 2: `pillar2_contagion/model.py` (simple GCN or GraphSAGE), `graph_data.py` (synthetic or small public graph, partitioned into 2–3 subgraphs). `client.py` and `run_train.py` stubs that run 2 rounds with 2 clients. No full evaluation yet.
  - Pillar 3: `pillar3_fedrl/env.py` (simple discrete state/action env: state = [price_deviation, volatility], action = {no-op, circuit_breaker}, reward = -|deviation|). `policy.py` (small MLP policy). `client.py` stub: collect 10 steps, compute simple policy gradient, return. Server stub. Run 2 rounds.
- **Acceptance:** `train_pillar2.py` and `train_pillar3.py` run without error for 2 rounds; no requirement on performance yet.

#### Q1.7 — Privacy integration points (Week 6)

- **Deliverables:**
  - `fedstable/privacy/dp.py`: function `apply_dp_gradient(grad, clip_norm, noise_scale)` and optional wrapper for client training loop (clip then add noise before send). Config: `privacy.dp.enabled`, `clip_norm`, `noise_scale`.
  - `fedstable/privacy/secure_agg.py`: interface only (e.g., `mask_update(update, cid, rnd)` returning update as-is). Document in README.
- **Acceptance:** With `privacy.dp.enabled=true`, Pillar 1 training runs and metrics are logged; optional plot of epsilon estimate vs accuracy.

### 14.3 Q2 Milestones (Detailed)

#### Q2.1 — Pillar 2 complete (Weeks 1–3)

- **Deliverables:**
  - Graph data: `graph_sources.py` builds a DeFi-style graph from DeFi Llama (or synthetic) with node features and edge indices. Partition into K silos (e.g., by protocol type or random edge split). Save format as in Section 9.4.
  - Full GNN training: `pillar2_contagion/client.py` and server with FedAvg; `run_train.py` and config for rounds, local epochs. Centralized GNN baseline on full graph.
  - Evaluation: `evaluate_pillar2.py` — correlation with centralized “oracle” contagion score or node-ranking metrics. Document in README.
- **Acceptance:** Pillar 2 runs end-to-end; federated vs centralized comparison documented; metrics in artifacts.

#### Q2.2 — Pillar 3 complete (Weeks 2–4)

- **Deliverables:**
  - Simulator: `env.py` with state space, action space, and reward aligned with depeg/contagion (use Pillar 2 score as state component if ready). Multiple env instances = different parameters (heterogeneous clients).
  - FedRL algorithm: Implement FedSVRPG or simple FedPG (policy gradient aggregation). Each client: collect trajectories, compute gradient, send. Server: aggregate, broadcast new policy.
  - Evaluation: Average return over evaluation episodes; depeg frequency. Compare FedRL vs local-only vs no intervention.
- **Acceptance:** Pillar 3 runs end-to-end; results in README; checkpoint and metrics saved.

#### Q2.3 — Cross-pillar and robustness (Weeks 3–4)

- **Deliverables:**
  - Optional: Pillar 1 uses “contagion score” from Pillar 2 as an extra feature (offline: precompute Pillar 2 scores; feed into Pillar 1 dataset). Document in README.
  - Non-IID robustness: Add a silo config where one client has only “normal” period and another “crisis” period; report performance. Optional: FedProx or personalization (per-client head) and compare.
  - Communication cost: Log bytes per round and total; add to metrics and README.
- **Acceptance:** README includes “Robustness” and “Communication” subsections with results.

#### Q2.4 — Reproducibility and packaging (Weeks 4–5)

- **Deliverables:**
  - All experiments reproducible from config + seed. `scripts/run_all_baselines.sh` (or equivalent) that runs Pillar 1/2/3 with default configs and saves artifacts.
  - `requirements.txt` / `pyproject.toml` updated and pinned. Optional: Dockerfile for running training in container.
  - README: “Reproducibility” section with exact commands and expected outcomes.
- **Acceptance:** New clone + install + run script produces comparable metrics (within reported variance).

#### Q2.5 — Tech pitch and documentation (Weeks 5–6)

- **Deliverables:**
  - **Tech pitch:** Document or deck (`docs/PITCH.md` or `docs/PITCH.pdf`) covering: problem (silos, stablecoin/DeFi risk), approach (three pillars), architecture (Flower, PyTorch, data flow), main results (tables: Fed vs central vs local; communication; optional DP), and “what’s next” (production, secure agg, real partners). No marketing fluff; technical only.
  - README: Final pass — ensure all sections (data, configs, run instructions, evaluation, references) are complete. Add “Quick start” at top with 3–5 commands.
- **Acceptance:** Pitch is self-contained and technically accurate; README allows a new contributor to run and extend the project.

### 14.4 Pitch Content (Tech-Only)

The Q2 pitch should include:

- **Problem:** Stablecoin depeg and DeFi contagion require cross-institutional data; data cannot be centralized.
- **Solution:** Federated learning for (1) depeg prediction, (2) contagion/systemic risk, (3) intervention policy learning.
- **Architecture:** Flower server + PyTorch clients; silo simulation from public data; optional DP and secure-agg interface.
- **Results:** Tables — Pillar 1: FedAvg vs centralized vs local-only (AUC, Brier); Pillar 2: Fed GNN vs centralized GNN (e.g., correlation or NDCG); Pillar 3: FedRL vs baselines (return, depeg rate). Optional: communication cost, DP (epsilon vs accuracy).
- **Code and reproducibility:** GitHub link; commands to reproduce; license.
- **Next steps:** Real data partnerships, secure aggregation implementation, production API, and possible publication targets (workshop or conference).

---

## 15. Testing Strategy

### 15.1 Unit Tests

- **Core:** `set_seed` reproducibility (two runs same seed → same random values). Config load and override.
- **Data:** `silo_partition` with known table produces correct splits and no overlap.
- **Models:** Pillar 1 model forward shape; Pillar 2 GNN forward on small graph; Pillar 3 env step and reward range.
- **Privacy:** `apply_dp_gradient` reduces gradient norm when clipping and changes values when noise > 0.

### 15.2 Integration Tests

- **FL:** 2 clients, 2 rounds; after aggregation, global model parameters change; both clients receive same global model in round 2. Use tiny data and small model so test is fast.
- **E2E (optional):** One full training run per pillar with minimal config (1 round, 2 clients, 10 samples each) and assert metrics file exists and contains expected keys.

### 15.3 Test Data

- Synthetic: small DataFrames and small graphs generated in `tests/conftest.py`. No real user data. Optional: 1–2 real public files (e.g., one day of prices) for smoke test only; document in tests README.

---

## 16. Deployment and Operations

### 16.1 Simulation vs Production

- Q1–Q2 target is **simulation:** all clients run on one machine (or a few processes). Production would require: separate client deployments per institution, authenticated and encrypted channels, and possibly secure aggregation. This spec does not define production deployment; only interfaces (e.g., client API, config) are designed so that productionization is feasible later.

### 16.2 Inference Service (Optional)

- If REST API is implemented: run with `uvicorn fedstable.api.app:app`. Model loaded from env `MODEL_PATH` or config. Recommend running behind reverse proxy and rate limiting in production; auth not specified here.

### 16.3 Resource Requirements

- **Minimum (simulation):** 8 GB RAM, 4 CPU cores, no GPU required for small models. GPU recommended for larger GNN or Transformer. Disk: space for raw and processed data (e.g., 1–10 GB depending on sources).
- **Scaling:** More clients or larger models may require more memory and communication; document observed usage in README after first runs.

---

## 17. References and Prior Art

### 17.1 Federated Learning

- McMahan et al., “Communication-Efficient Learning of Deep Networks from Decentralized Data” (FedAvg), AISTATS 2017.
- Li et al., “Federated Optimization in Heterogeneous Networks” (FedProx), MLSys 2020.
- Flower documentation: https://flower.ai/docs/.

### 17.2 Federated Learning and Finance

- NeurIPS 2025 workshop: “Privacy-Preserving Financial Fraud Detection” (generative models, federated boosting).
- Starlit: “Privacy-Preserving Federated Learning to Enhance Financial Fraud Detection,” eprint 2024/090.
- Fed-RD: “Privacy-Preserving Federated Learning for Financial Crime Detection,” IEEE.

### 17.3 Stablecoin and DeFi Risk

- “Stablecoin Depegging Risk Prediction,” ScienceDirect / SSRN 2024.
- “Collapse of Silicon Valley Bank and USDC Depegging: A Machine Learning Experiment,” MDPI.
- BIS Working Papers: “Public information and stablecoin runs”; “Reserve management and run risk.”
- “Systemic Risk in DeFi: A Network-Based Fragility Analysis of TVL Dynamics,” arXiv.
- DeXposure-FM: “Time-Series Graph Foundation Model for DeFi Credit Exposure,” arXiv.

### 17.4 Federated Graph and FedRL

- “Federated Graph Learning” (survey and methods).
- “FedMRL: Data Heterogeneity Aware Federated Multi-agent Deep Reinforcement Learning for Medical Imaging,” MICCAI 2024.
- “Momentum for the Win: Collaborative Federated Reinforcement Learning across Heterogeneous Environments,” MLR 2024.

### 17.5 Blockchain and FL

- FLock: “Robust and Privacy-Preserving Federated Learning based on Practical Blockchain State Channels,” eprint 2024/1797.
- DP-BBVFL: “Differentially Private Blockchain-Based Vertical Federated Learning,” 2024.

---

## 18. Glossary

- **Depeg:** Stablecoin market price deviating significantly from its peg (e.g., $1 for USD-backed).
- **Silo:** A single institution’s data partition (one client in FL).
- **FedAvg:** Federated Averaging — server aggregates client model weights by weighted average.
- **Non-IID:** Data across clients not independent and identically distributed.
- **DP (Differential Privacy):** Formal privacy guarantee; (epsilon, delta)-DP.
- **Secure aggregation:** Cryptographic protocol so server learns only the sum of client updates, not individual updates.
- **Contagion (DeFi):** Propagation of stress or default across protocols via shared collateral, liquidity, or dependencies.
- **Systemic importance:** Measure of a node’s (e.g., protocol’s) contribution to system-wide risk.
- **FedRL:** Federated Reinforcement Learning — RL with multiple clients learning a shared or personalized policy without sharing raw trajectories.
- **Reserve stress:** Proxy for risk that reserve assets cannot meet redemptions (liquidity or solvency concern).

---

## 19. Data Schemas and Column Definitions

### 19.1 Pillar 1 — Depeg Features (Parquet/CSV)

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime64[ns] | Observation time (UTC). |
| `client_id` | str or int | Silo identifier (e.g. `exchange_a`, `0`). |
| `stablecoin` | str | Symbol (e.g. USDC, USDT, DAI). |
| `price` | float64 | Market price. |
| `peg_deviation_pct` | float64 | (price - peg) / peg * 100. |
| `volume_24h` | float64 | Rolling 24h volume. |
| `volatility_24h` | float64 | Rolling 24h std of returns. |
| `supply` | float64 | Circulating supply (if available). |
| `reserve_proxy` | float64 | Optional reserve health proxy (e.g. 0–1). |
| `label_depeg_1h` | int (0/1) | 1 if depeg in next 1h. |
| `label_depeg_6h` | int (0/1) | 1 if depeg in next 6h. |
| `label_depeg_24h` | int (0/1) | 1 if depeg in next 24h. |

Optional: `contagion_score` (float) once Pillar 2 is available. Window features (e.g. `peg_deviation_lag_1`, `volume_change`) can be added in preprocessing. Schema validation: `fedstable/data/schemas.py` can expose a `validate_depeg_schema(df)` helper.

### 19.2 Pillar 2 — Graph (Per-Silo)

- **Node features:** NumPy array or PyG `x` of shape `[num_nodes, num_features]`. Features: TVL (normalized), volume, protocol category encoding.
- **Edge index:** `[2, num_edges]` in COO format; node indices local to that silo or global (document which).
- **Edge weights (optional):** Float tensor of shape `[num_edges]` (e.g. exposure amount).
- **Labels (optional):** Node-level contagion or systemic-importance score for supervised training; shape `[num_nodes]`. If unavailable, use self-supervised or link-prediction objective.
- **Storage:** One file per silo, e.g. `data/processed/graph_silo_0.pt` containing a PyG `Data` object or dict with `x`, `edge_index`, `y`.

### 19.3 Pillar 3 — Simulator State

- **State vector (example):** `[peg_deviation, volatility_proxy, collateral_ratio_proxy, contagion_proxy]` — all floats, normalized. Length and order fixed in `env.py`.
- **Action space:** Discrete: `{0: no_op, 1: circuit_breaker}` or extended; document in `pillar3_fedrl/env.py`.
- **Reward:** Float; e.g. `-abs(peg_deviation) - 0.1 * intervention_penalty`. No file format; generated at runtime.

---

## 20. Configuration Reference (Full Example)

### 20.1 base.yaml

```yaml
seed: 42
device: cuda  # or cpu
data_root: ./data
processed_dir: ${data_root}/processed
raw_dir: ${data_root}/raw
artifacts_dir: ./artifacts
log_level: INFO
```

### 20.2 pillar1.yaml (Full)

```yaml
defaults:
  - base

pillar: 1

model:
  type: lstm  # or transformer
  input_size: 8
  hidden_size: 64
  num_layers: 2
  dropout: 0.2
  num_horizons: 3  # 1h, 6h, 24h

training:
  rounds: 50
  clients_per_round: 3
  local_epochs: 2
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam

data:
  silo_config: configs/silos_3.yaml
  sequence_length: 24
  horizon: 6  # primary label: 6h
  train_ratio: 0.7
  val_ratio: 0.1

evaluation:
  central_eval_every: 5
  metrics: [auc_roc, auc_pr, brier_score]

privacy:
  dp:
    enabled: false
    clip_norm: 1.0
    noise_scale: 0.01
```

### 20.3 silos_3.yaml (Example)

```yaml
partition_strategy: by_stablecoin  # or random, by_exchange
num_silos: 3
silos:
  - id: 0
    name: silo_usdc
    filter: { stablecoin: USDC }
  - id: 1
    name: silo_usdt
    filter: { stablecoin: USDT }
  - id: 2
    name: silo_dai
    filter: { stablecoin: DAI }
```

### 20.4 pillar2.yaml (Minimal)

```yaml
defaults:
  - base

pillar: 2

model:
  type: gcn  # or graphsage
  in_channels: 8
  hidden_channels: 32
  out_channels: 1
  num_layers: 2

training:
  rounds: 30
  clients_per_round: 2
  local_epochs: 3
  learning_rate: 0.01

data:
  graph_global_path: ${processed_dir}/graph_global.pt
  silo_config: configs/silos_graph_2.yaml
```

### 20.5 pillar3.yaml (Minimal)

```yaml
defaults:
  - base

pillar: 3

env:
  state_dim: 4
  action_dim: 2
  max_steps: 100
  reward_scale: 1.0

policy:
  hidden_sizes: [64, 64]
  lr: 1e-3

training:
  rounds: 20
  clients_per_round: 2
  steps_per_round: 500
```

---

## 21. API Request/Response Schemas (Optional REST)

### 21.1 POST /predict (Depeg)

**Request:**

```json
{
  "features": [[0.1, 0.02, 1.5, 0.01, 0.0, 0.0, 0.0, 0.0]],
  "model": "depeg",
  "horizon": "6h"
}
```

`features`: list of feature vectors; each vector length must match model `input_size` (e.g. 8). Order of features must match training (see Section 19.1).

**Response:**

```json
{
  "probabilities": [0.02],
  "model_version": "pillar1_v1",
  "horizon": "6h"
}
```

### 21.2 POST /health

**Response:** `{"status": "ok", "model_loaded": true}`. Used for readiness checks.

### 21.3 Error Response (4xx/5xx)

```json
{
  "detail": "Invalid feature dimension: expected 8, got 5",
  "code": "VALIDATION_ERROR"
}
```

---

## 22. Integration Checklist (External Systems)

### 22.1 Flower (flwr)

- [ ] Install: `pip install flwr[simulation]` or `flwr`.
- [ ] Version: 1.25+ for Python 3.10+.
- [ ] Use `flwr.client.Client`, `flwr.client.NumPyClient`, or `flwr.client.ClientApp`; implement `fit` and `evaluate` where needed.
- [ ] Server: `flwr.server.Server`, `flwr.server.strategy.FedAvg`; optional custom `Strategy` subclass for FedProx or DP aggregation.
- [ ] Simulation: `flwr.simulation.run_simulation` or manual loop with `start_client`/`start_server` in separate processes.

### 22.2 PyTorch Geometric

- [ ] Install: `pip install torch_geometric`; follow official instructions for PyTorch version and optional CUDA.
- [ ] Use `torch_geometric.data.Data` for graph objects; `DataLoader` with `batch` for batching multiple graphs if needed (Pillar 2 may use single graph per client).
- [ ] Model: `torch_geometric.nn.GCNConv`, `GraphSAGE`, or custom MessagePassing.

### 22.3 Data Sources (On-Chain / APIs)

- [ ] CoinGecko (or similar): endpoint for stablecoin price, volume, supply; rate limits and API key if required. Adapter in `data/sources/coingecko.py`.
- [ ] DeFi Llama: TVL and protocol list; adapter in `data/sources/defillama.py`.
- [ ] Depeg events: manual or script-generated `depeg_events.json` with `date`, `stablecoin`, `severity`, `duration_hours`.

### 22.4 Differential Privacy (Opacus)

- [ ] Install: `pip install opacus`. Compatibility: check PyTorch version.
- [ ] Use `opacus.privacy_engine.PrivacyEngine` for per-sample gradient clipping and noise; attach to optimizer. Alternatively implement minimal DP-SGD in `fedstable/privacy/dp.py` (clip by norm, add Gaussian noise) to avoid dependency.

### 22.5 Experiment Tracking (Optional)

- [ ] MLflow: log params, metrics, artifacts in `run_train.py`; set `MLFLOW_TRACKING_URI` in env.
- [ ] Weights & Biases: `wandb.init`, `wandb.log` in training loop; optional `wandb.config.update(config)`.

---

## 23. Development Workflow

### 23.1 Branching

- `main`: stable; all tests pass; README and configs reflect current state.
- Feature branches: `feature/pillar1-dp`, `feature/pillar2-gnn`, etc. Merge after review or self-review and passing tests.

### 23.2 Commit Conventions

- Prefer descriptive messages: `feat(pillar1): add LSTM model and dataset`; `fix(silo): correct partition when filter is empty`.
- Scope: `pillar1`, `pillar2`, `pillar3`, `data`, `privacy`, `docs`, `tests`.

### 23.3 Adding a New Silo Strategy

1. Extend `fedstable/data/silo_partition.py`: add a new branch in partition logic (e.g. `by_venue`).
2. Document in `configs/` with an example `silos_*.yaml`.
3. Update `data/README.md` and Section 9.3 of this spec.
4. Add test in `tests/test_silo_partition.py` for the new strategy.

### 23.4 Adding a New FL Strategy (e.g. FedProx)

1. Implement or use Flower `Strategy` that applies proximal term in client training (or aggregate with different weights).
2. Add config key `training.strategy: fedprox` and `training.mu: 0.01` (proximal coefficient).
3. Wire in `pillar1_depeg/run_train.py` (and optionally pillar2/3).
4. Document in README and add baseline comparison in Q2.

### 23.5 Adding a New Metric

1. Implement in `fedstable/evaluation/metrics.py` (e.g. `def calibration_error(y_true, y_pred, n_bins=10)`).
2. Call from `evaluate_pillar1.py` (or pillar2/3); add to returned dict.
3. Add to config `evaluation.metrics` and document in Section 5.5 / 6.5 / 7.4.

---

## 24. Hyperparameter Tables (Quick Reference)

### 24.1 Pillar 1

| Parameter | Typical range | Notes |
|-----------|----------------|-------|
| `rounds` | 30–100 | More rounds for more clients or non-IID. |
| `clients_per_round` | 2–5 | All clients or subset. |
| `local_epochs` | 1–5 | 1 = FedAvg; >1 = multiple local passes. |
| `batch_size` | 16–64 | Smaller for small silos. |
| `learning_rate` | 1e-4 – 1e-2 | Often 1e-3. |
| `sequence_length` | 12–168 | History length (e.g. 24 = 24h at 1h freq). |
| `hidden_size` | 32–128 | Model capacity. |
| `clip_norm` (DP) | 0.5–2.0 | Lower = more privacy, less utility. |
| `noise_scale` (DP) | 0.01–0.1 | Higher = more privacy. |

### 24.2 Pillar 2

| Parameter | Typical range | Notes |
|-----------|----------------|-------|
| `rounds` | 20–50 | | 
| `local_epochs` | 2–5 | |
| `in_channels` | 4–16 | Node feature dim. |
| `hidden_channels` | 16–64 | |
| `num_layers` | 2–3 | GNN depth. |

### 24.3 Pillar 3

| Parameter | Typical range | Notes |
|-----------|----------------|-------|
| `rounds` | 10–30 | |
| `steps_per_round` | 200–1000 | Trajectory steps per client per round. |
| `policy lr` | 1e-4 – 1e-2 | |
| `gamma` (discount) | 0.95–0.99 | |
| `max_steps` (env) | 50–200 | Episode length. |

---

## 25. Troubleshooting (Technical)

### 25.1 Training Loss Not Decreasing (Pillar 1)

- Check label balance (depeg events rare → use class weights or oversampling).
- Verify feature scaling (normalize or standardize).
- Increase `local_epochs` or `learning_rate`; ensure clients have enough samples per round.
- Run centralized baseline first to confirm data and model are learnable.

### 25.2 Federated Performance Much Worse Than Centralized

- Expected under non-IID; try FedProx (small `mu`) or personalization (per-client head).
- Increase rounds; ensure `clients_per_round` includes all clients sometimes.
- Check silo partition (e.g. one silo has only one class) and consider different partition for robustness experiments.

### 25.3 Out of Memory (Pillar 2)

- Reduce batch size (if batching graphs) or use smaller subgraphs per client.
- Use gradient checkpointing or fewer GNN layers.
- Run on CPU with smaller `hidden_channels`.

### 25.4 Flower Connection Errors

- In simulation, ensure server starts before clients; use `flwr.simulation.run_simulation` for coordinated start.
- For real distributed: check firewall and that server address/port are correct; use TLS in production.

### 25.5 DP: Accuracy Drops Too Much

- Increase `clip_norm` or decrease `noise_scale` (reduces privacy).
- Use more rounds so model can converge with noisier updates.
- Report epsilon/delta and consider privacy accounting (e.g. Opacus or tensorflow_privacy).

### 25.6 Reproducibility: Different Runs Give Different Results

- Set seed in config and call `set_seed(config.seed)` at start of every run.
- Pin library versions; check that PyTorch/CUDA are deterministic where needed (`torch.backends.cudnn.deterministic = True` if applicable).
- Document “expected variance” for FL (e.g. ±2% AUC across 5 seeds).

---

## 26. Quick Start (Commands)

After cloning and installing dependencies:

```bash
# 1. Create data dirs
mkdir -p data/raw data/processed artifacts

# 2. Download raw data (implement script)
python scripts/download_data.py --config configs/base.yaml

# 3. Preprocess and partition into silos
python scripts/preprocess.py --config configs/pillar1.yaml

# 4. Train Pillar 1 (federated)
python scripts/train_pillar1.py --config configs/pillar1.yaml

# 5. Evaluate
python scripts/evaluate_pillar1.py --checkpoint artifacts/pillar1/run_<timestamp>/model.pt --config configs/pillar1.yaml
```

Pillar 2 and 3:

```bash
python scripts/train_pillar2.py --config configs/pillar2.yaml
python scripts/train_pillar3.py --config configs/pillar3.yaml
```

Run all baselines (Q2):

```bash
bash scripts/run_all_baselines.sh
```

---

## 27. License and Citation (Placeholder)

- **License:** TBD (e.g. Apache 2.0 or MIT). Stated in repository.
- **Citation:** If you use FedStable in research, please cite this repository and any accompanying paper (to be added when published).

---

**End of specification.** Sections 1–27. For updates, maintain this README and bump a version or date in the document header.
