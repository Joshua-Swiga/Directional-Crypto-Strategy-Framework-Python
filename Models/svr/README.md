# Support Vector Regression (SVR) Model

## Purpose
Model non-linear regression with kernel trick.

## Justification

- **Non-linear effects**: Good for datasets where features have non-linear effects on target.
- **Robust to outliers**: Robust to outliers due to ε-insensitive loss.
- **Small datasets**: Works well with small to medium datasets (979 rows is fine).

## Relevance

- **Returns prediction**: Can be applied to `future_returns` to capture small-scale fluctuations.
- **Independent models**: Separate SVR models can handle close vs returns prediction independently.
- **Kernel flexibility**: RBF kernel captures complex non-linear patterns.

## Hyperparameters

- `kernel`: Kernel type ('rbf', 'linear', 'poly') - default: 'rbf'
- `C`: Regularization parameter (default: 1.0)
- `epsilon`: Epsilon-tube width (default: 0.1)
- `gamma`: Kernel coefficient for 'rbf' (default: 'scale')

## Usage

```python
from models.svr.train import train_model
from models.svr.predict import predict

# Train
model_close, model_returns = train_model()

# Predict
predictions_close = predict(model_close, X_new, target='close')
predictions_returns = predict(model_returns, X_new, target='returns')
```
