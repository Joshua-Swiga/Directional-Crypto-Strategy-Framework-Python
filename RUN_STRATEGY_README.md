# Central Entry Point — run_strategy.py

`run_strategy.py` is the central entry point for the trading strategy pipeline. It runs analysis, trains models, evaluates them, and generates reports.

## Quick Start

```bash
# Full pipeline (analysis → train → evaluate → report)
python run_strategy.py

# Or step by step:
python run_strategy.py --analysis          # Exploratory analysis & correlation
python run_strategy.py --train            # Train all models
python run_strategy.py --evaluate         # Evaluate all models
python run_strategy.py --report           # Generate evaluation report
```

## One-run report policy (auto-clean)

By default, each run of `run_strategy.py` will:

- Delete all existing files in `reports/`
- Delete all existing files in `correlation_images/`
- Generate a single fresh report set named `evaluation_report_latest.*`

If you want to keep history, run with:

```bash
python run_strategy.py --keep-history
```

## Pipeline Steps

| Step | What it does |
|------|--------------|
| **Data** (`--data`) | Runs `data_collection_and_cleaning.deriving_additional_features()` |
| **Analysis** (`--analysis`) | Runs `Exploratory_Analysis_Correlation` → creates `ml_data/` |
| **Train** (`--train`) | Trains all (or selected) models |
| **Evaluate** (`--evaluate`) | Evaluates models on test set |
| **Report** (`--report`) | Writes evaluation report to `reports/` |

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--analysis` | Run exploratory analysis & correlation only |
| `--data` | Run data collection & feature derivation first |
| `--train` | Train models |
| `--evaluate` | Evaluate models |
| `--report` | Generate evaluation report |
| `--models M1 M2` | Train/evaluate only these models |
| `--skip-analysis` | Skip analysis (use existing `ml_data/`). Use with `--train`. |
| `--backtest` | Run an MVP backtest and write `backtest_latest.*` to `reports/`. |
| `--bt-model` | Which model to backtest (default: `linear_regression`). |
| `--bt-threshold` | Only trade when \(|predicted return| > threshold\). |
| `--bt-fee-bps` | Trading fee in bps (default: 5). |
| `--bt-slippage-bps` | Slippage in bps (default: 2). |
| `--bt-no-short` | Long/flat only (no shorts). |

## Examples

```bash
# Full pipeline (default)
python run_strategy.py

# Only analysis (creates ml_data for first time)
python run_strategy.py --analysis

# Train specific models
python run_strategy.py --train --models linear_regression random_forest xgboost

# Skip analysis, train all models (assumes ml_data exists)
python run_strategy.py --skip-analysis --train

# Evaluate and generate report only
python run_strategy.py --evaluate --report

# Data + analysis (for fresh data)
python run_strategy.py --data --analysis
```

## Output

- **ml_data/** — Created by analysis step. Contains `X_train`, `X_test`, targets, feature metadata.
- **reports/** — One-run report artifacts:
  - `evaluation_report_latest.pdf` (clean, readable)
  - `evaluation_report_latest.html` (interactive sortable tables)
  - `evaluation_report_latest.md` / `.txt` / `.json`
  - `backtest_latest.pdf` (clean, readable)
  - `backtest_latest.html` (interactive trades table)
  - `backtest_latest_trades.csv` (trade log)
- **correlation_images/** — Correlation heatmaps from analysis.
- **Models/*/*** — Trained model files (`.pkl`) in each model directory.

## Report Contents

Each report includes:

- Summary table: Model, Target, RMSE, MAE, R², MAPE
- Per-model details for close and returns prediction
- Timestamp of generation

## Prerequisites

1. Run `--analysis` at least once (or have `ml_data/` from a previous run).
2. For `--data`: raw data file must exist (run `getting_data` or `data_collection_and_cleaning` data fetch first).
3. Dependencies: `scikit-learn`, `pandas`, `numpy`, `xgboost` (for XGBoost), `statsmodels` (for ARIMA).
