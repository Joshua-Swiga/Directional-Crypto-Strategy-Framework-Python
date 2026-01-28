# Crypto Regression + Pattern Trading Framework

**Predict high-probability directional moves in crypto using statistics + patterns — built for home traders.**

This repository implements a complete, modular trading framework focused on **predicting whether price is more likely to go up or down** over the next ~15 candles (≈2.5 days on 4H timeframe). It combines:

- **Linear & multiple regression** for trend identification and price prediction
- **Correlation analysis & statistical significance testing** (p-values, R²) for feature selection
- **Apriori algorithm** for mining high-confidence candlestick patterns
- **Full risk & capital management** (Sharpe/Sortino, max drawdown, position sizing, margin control)

Designed for **Bitcoin (BTC), Ethereum (ETH), Solana (SOL)** and similar high-liquidity cryptocurrencies on 4H candles. Optimized for low-frequency, home-based trading with minimal screen time.

## Why This Framework?
- Emphasizes **statistical rigor** over black-box ML (easy to understand & debug)
- Multi-layer confirmation: regression signal + pattern validation
- Strong focus on **risk-adjusted performance** (Sharpe, Sortino, Calmar, Profit Factor)
- Continuous improvement loop: weekly model retraining & KPI tracking
- No repainting — signals generated only on closed candles

## Key Features
- Data preprocessing: OHLCV, derived features (body size, range, volatility, rolling volume)
- Exploratory analysis: correlation matrices, pairwise & multiple regression
- Pattern mining: Apriori for frequent bullish/bearish candle sequences
- Signal generation: directional bias + optional pattern filter
- Risk engine: ATR-based stops, volatility-adjusted sizing, max 30–40% margin exposure
- Backtesting & evaluation: returns, win rate, drawdown, risk-adjusted metrics
- Optional: Monte Carlo simulations, macro factor integration

## Timeframe Recommendation
**4H candles** — optimal balance of noise reduction, statistical reliability, and low monitoring (check 4–6× per day).

## Getting Started

### Prerequisites
- Python 3.8+
- Libraries: pandas, numpy, statsmodels, mlxtend (for Apriori), matplotlib/seaborn, scikit-learn (optional extensions)
- Data source: CCXT, Binance API, CoinGecko, or CSV files

### Installation
```bash
git clone https://github.com/YOUR-USERNAME/Crypto-Regression-Pattern-Trader.git
cd Crypto-Regression-Pattern-Trader
pip install -r requirements.txt
