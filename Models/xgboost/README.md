# XGBoost / Gradient Boosting Model

## Purpose
Advanced tree-based ensemble for high-accuracy regression.

## Justification

- **Sequential error correction**: Boosting sequentially corrects errors, making it ideal for financial datasets with mixed signal strengths.
- **Non-linearities**: Handles non-linearities and interactions better than Random Forest for small to medium-sized datasets.
- **Efficient**: Works well with 7 selected features, no need to expand excessively.

## Relevance

- **Returns prediction**: Perfect for `future_returns`, which have weak but relevant signals (`candle_body`, `quote_volume`).
- **Independent models**: Can be trained independently for `future_close` and `future_returns`.
- **High accuracy**: Often outperforms Random Forest on structured data.

## Hyperparameters

- `n_estimators`: Number of boosting rounds (default: 100)
- `max_depth`: Maximum tree depth (default: 6)
- `learning_rate`: Step size shrinkage (default: 0.1)
- `subsample`: Fraction of samples for training (default: 0.8)

## Usage

```python
from models.xgboost.train import train_model
from models.xgboost.predict import predict

# Train
model_close, model_returns = train_model()

# Predict
predictions_close = predict(model_close, X_new, target='close')
predictions_returns = predict(model_returns, X_new, target='returns')
```
