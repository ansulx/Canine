# Data directory

- **raw/**: Raw downloads (CoinGecko, DeFi Llama, depeg_events.json). Populated by `scripts/download_data.py`.
- **processed/**: Preprocessed tables and graph; per-silo subdirs (e.g. `silo_0/train.parquet`). Populated by `scripts/preprocess.py`.

See main README Section 9 (Data Pipeline) and Section 19 (Data Schemas) for column definitions and formats.
