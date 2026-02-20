# ARIMA / SARIMAX Model

## Purpose
Classical time series forecasting of continuous price levels.

## Justification

- **Time series persistence**: Future close is highly persistent (R² ~1.0), meaning past values can strongly predict the next value.
- **Exogenous features**: Can incorporate exogenous features (`atr`, `num_trades`, `volatility`) via SARIMAX.
- **Sequential patterns**: Captures autoregressive and moving average components in price series.

## Relevance

- **Level prediction**: Ideal if you want a separate forecast model based purely on sequential data, giving another independent prediction to ensemble with ML models.
- **Complementary**: Works alongside ML models to provide time-series-specific insights.
- **Baseline**: Good baseline for time-series forecasting.

## Model Types

- **ARIMA**: AutoRegressive Integrated Moving Average (univariate)
- **SARIMAX**: Seasonal ARIMA with eXogenous regressors (multivariate)

## Hyperparameters

- `order`: (p, d, q) - ARIMA order
  - p: AR order (default: 2)
  - d: Differencing order (default: 1)
  - q: MA order (default: 2)
- `seasonal_order`: (P, D, Q, s) - Seasonal order (optional)
- `exog`: Exogenous variables (features)

## Usage

```python
from models.arima.train import train_model
from models.arima.predict import predict

# Train
model_close = train_model()

# Predict
predictions_close = predict(model_close, n_periods=10)
```

## Note

ARIMA models are univariate by nature. For `future_returns`, consider using differenced close prices or returns directly.
