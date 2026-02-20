# Random Forest Regressor Model

## Purpose
Predict non-linear relationships in both `future_close` and `future_returns`.

## Justification

- **Non-linear interactions**: Handles non-linear feature interactions naturally (important for returns where correlations are weaker).
- **Overfitting resistance**: Resistant to overfitting if you tune trees, depth, and number of estimators.
- **Feature importance**: Can output feature importances, helping to verify which features drive returns.

## Relevance

- **Returns prediction**: Returns (`future_returns`) are more volatile and weakly correlated with features; Random Forest can capture subtle patterns that linear regression misses.
- **Independent predictions**: Can be applied separately for `future_close` and `future_returns` to provide independent predictions.
- **Robustness**: Less sensitive to outliers than linear models.

## Hyperparameters

- `n_estimators`: Number of trees (default: 100)
- `max_depth`: Maximum depth of trees (default: 10)
- `min_samples_split`: Minimum samples to split (default: 2)
- `min_samples_leaf`: Minimum samples in leaf (default: 1)

## Usage

```python
from models.random_forest.train import train_model
from models.random_forest.predict import predict

# Train
model_close, model_returns = train_model()

# Predict
predictions_close = predict(model_close, X_new, target='close')
predictions_returns = predict(model_returns, X_new, target='returns')
```
