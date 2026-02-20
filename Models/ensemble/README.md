# Ensemble / Hybrid Model

## Purpose
Combine multiple models to get robust predictions.

## Justification

- **Model diversity**: E.g., Linear Regression for trend (future_close) + Random Forest / XGBoost for volatility (future_returns).
- **Reduced overfitting**: Reduces risk of overfitting to one model type.
- **Robust predictions**: Provides isolated but complementary outputs for each target variable.

## Relevance

- **Robustness**: Makes predictions more robust in volatile markets.
- **Complementary**: Each model type captures different patterns:
  - Linear Regression: Strong linear trends
  - Random Forest/XGBoost: Non-linear interactions
  - SVR: Non-linear kernel patterns
  - Neural Network: Complex feature interactions
  - ARIMA: Time-series patterns
- **Trading strategy**: Can use weighted ensemble or voting to combine predictions.

## Ensemble Methods

1. **Averaging**: Simple average of predictions
2. **Weighted Average**: Weight by validation performance
3. **Stacking**: Meta-learner combines base models
4. **Voting**: For classification (if converting to buy/sell signals)

## Usage

```python
from models.ensemble.train import train_ensemble
from models.ensemble.predict import predict_ensemble

# Train ensemble
ensemble = train_ensemble()

# Predict
predictions_close, predictions_returns = predict_ensemble(X_new)
```

## Configuration

Edit `config.py` to select which models to include in the ensemble and their weights.
