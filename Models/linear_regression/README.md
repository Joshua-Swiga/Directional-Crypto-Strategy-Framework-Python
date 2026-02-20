# Linear Regression Model

## Purpose
Predict `future_close` or `future_returns` as a continuous variable.

## Justification

- **Excellent baseline performance**: Your linear regression baseline already shows excellent R² for future_close (~1.0) and high R² for future_returns (~0.931).
- **Strong linear relationships**: Linear relationships dominate for level prediction (`future_close`) because features like `open`, `atr`, `num_trades`, and `volatility` show strong linear correlation with future_close.
- **Interpretability**: Easy to interpret coefficients, which is good for understanding feature importance and debugging.

## Relevance

- **Benchmark model**: Acts as a benchmark for other models.
- **Level prediction**: Perfect for predicting exact price levels (`future_close`) where linear persistence is strong.
- **Fast inference**: Very fast training and prediction, suitable for real-time trading.

## Usage

```python
from models.linear_regression.train import train_model
from models.linear_regression.predict import predict

# Train
model_close, model_returns = train_model()

# Predict
predictions_close = predict(model_close, X_new, target='close')
predictions_returns = predict(model_returns, X_new, target='returns')
```

## Model Files

- `train.py`: Training script
- `predict.py`: Prediction script
- `evaluate.py`: Evaluation script
- `model.py`: Model class definition
