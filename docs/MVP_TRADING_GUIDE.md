# MVP Trading Guide (Models + Backtest)

This guide explains how to run the pipeline and how the MVP turns predictions into trades.

## What this MVP does

- Builds features from Binance OHLCV
- Runs correlation + feature selection
- Trains ML models that predict:
  - `future_close` (next bar price level)
  - `future_returns` (next bar return)
- Backtests a simple strategy that trades based on predicted `future_returns`

## Quick start (recommended)

From `C:\Users\JoshuaSwiga\Desktop\strat`:

```powershell
# 1) Build features + ml_data/
python run_strategy.py --analysis

# 2) Train a model (start with linear regression)
python run_strategy.py --train --models linear_regression

# 3) Evaluate + generate report (PDF + HTML)
python run_strategy.py --evaluate --report --models linear_regression

# 4) Backtest (PDF + HTML + trades CSV)
python run_strategy.py --backtest --bt-model linear_regression
```

## Where outputs go (one-run policy)

Every run deletes old files in `reports/` and `correlation_images/` unless you pass `--keep-history`.

You will typically see:

- `reports/evaluation_report_latest.pdf`
- `reports/evaluation_report_latest.html` (interactive sortable tables)
- `reports/backtest_latest.pdf`
- `reports/backtest_latest.html` (interactive trades table)
- `reports/backtest_latest_trades.csv`
- `reports/backtest_latest_equity.png`

## How the MVP “gets into a trade”

The backtest uses predicted **next-bar returns**:

- Predict: \(\hat r_{t+1}\)
- Signal:
  - **LONG** if \(\hat r_{t+1} > threshold\)
  - **SHORT** if \(\hat r_{t+1} < -threshold\) (unless `--bt-no-short`)
  - **FLAT** otherwise
- PnL per bar (simplified):
  - `strategy_return = position * future_returns - costs`
  - costs are charged on **position changes** (turnover), using `fee_bps + slippage_bps`

### Reduce overtrading (recommended)

Use a threshold:

```powershell
python run_strategy.py --backtest --bt-threshold 0.0002
```

### Long-only (recommended for early MVP)

```powershell
python run_strategy.py --backtest --bt-no-short
```

## How to interpret the reports

### Evaluation report

Key metrics:

- **RMSE / MAE**: error size (lower is better)
- **R2**: explained variance (closer to 1 is better)
- **MAPE**: percent error (often misleading for returns; use RMSE/MAE/R2)

### Backtest report

Key metrics:

- **Total return**: growth of equity from 1.0
- **Max drawdown**: worst peak-to-trough decline
- **Sharpe**: risk-adjusted return (very high values can indicate leakage or unrealistically easy conditions)
- **Hit rate**: fraction of profitable bars
- **Turns**: number of position changes (impacts costs)

## Common mistakes

- Running models before `--analysis` (you need `ml_data/`)
- Using MAPE to judge return predictions (returns are near zero; MAPE explodes)
- Trusting a single holdout backtest (prefer walk-forward validation)

## Next steps to make it market-ready

- Walk-forward / rolling backtests across multiple windows
- Add risk management:
  - position sizing
  - max drawdown stop
  - volatility targeting
- Improve execution assumptions (spread, slippage, latency)

