# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Time-series glucose prediction using the REPLACE-BG dataset. Uses a CNN-LSTM model to predict future blood glucose levels from historical glucose, insulin bolus, and carbohydrate data. Built on PyTorch Lightning + Hydra configuration, based on nn-template v0.4.0.

## Common Commands

```bash
# Environment setup (using uv)
uv sync               # Install all dependencies
uv sync --extra dev   # Install with dev tools

# Training
uv run python src/glucose_prediction/run.py

# Training with overrides (Hydra)
uv run python src/glucose_prediction/run.py train.trainer.max_epochs=50 train.trainer.fast_dev_run=true

# Code quality
uv run pre-commit run --all-files

# Tests
uv run pytest -v
```

## Architecture

**Entry point:** `src/glucose_prediction/run.py` — Hydra-decorated `main()` that orchestrates the full train/test pipeline.

**Data flow:**
1. Raw CSV files per patient in `data/raw/patients/{patient_id}.csv`
2. `ReplaceBGDataset` (`data/replace_bg_dataset.py`) — extracts sliding windows of 16 timesteps (15-min intervals) with 3 features (glucose, bolus, carbs), applies z-normalization
3. `GlucoseDataModule` (`data/glucose_datamodule.py`) — splits patients 70/10/20 train/val/test, creates `ConcatDataset` per split
4. `GlucoseModule` (`pl_modules/glucose_module.py`) — Lightning module using L1 loss, RMSE metrics, denormalization with hardcoded stats (mean=160.875, std=63.601). Uses `input_length=12` history and `pred_length=4` horizon
5. `CNN_LSTM` (`modules/cnn_lstm.py`) — 4 Conv1d layers → LSTM (2 layers, hidden=100) → 3 FC layers. Supports auto-regressive prediction

**Configuration:** Hydra configs in `conf/` with hierarchy: `default.yaml` → `nn/`, `train/`, `hydra/` subdirectories. All components instantiated via `hydra.utils.instantiate()` with `_recursive_=False`.

**Logging:** Weights & Biases (configured in `conf/train/default.yaml`).

**Data preprocessing:** `src/glucose_prediction/preprocessing.ipynb` converts raw REPLACE-BG text files into per-patient CSVs.

## Code Style

- Black formatter, 120 char line length
- isort with black profile
- Flake8 linting (see `.flake8`)
- Google-style docstrings
- Pre-commit hooks enforce all formatting/linting on commit
