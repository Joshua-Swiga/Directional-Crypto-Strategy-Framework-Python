# Trading Strategy Models

This directory contains isolated model implementations for the trading strategy. Each model has its own directory with training, prediction, and evaluation scripts.

## Model Structure

Each model directory contains:
- `README.md`: Model explanation, justification, and usage
- `model.py`: Model class definition
- `train.py`: Training script
- `predict.py`: Prediction script
- `evaluate.py`: Evaluation script

## Available Models

### 1. Linear Regression (Baseline)
**Directory**: `linear_regression/`
- **Purpose**: Predict `future_close` or `future_returns` as continuous variables
- **Best for**: Level prediction (future_close) with strong linear persistence
- **Why**: Excellent R² for future_close, interpretable coefficients

### 2. Random Forest Regressor
**Directory**: `random_forest/`
- **Purpose**: Predict non-linear relationships
- **Best for**: Future returns with weak correlations
- **Why**: Handles non-linear interactions, feature importance, resistant to overfitting

### 3. XGBoost / Gradient Boosting
**Directory**: `xgboost/`
- **Purpose**: High-accuracy regression with sequential error correction
- **Best for**: Future returns with weak but relevant signals
- **Why**: Boosting corrects errors sequentially, handles non-linearities well

### 4. Support Vector Regression (SVR)
**Directory**: `svr/`
- **Purpose**: Non-linear regression with kernel trick
- **Best for**: Future returns with small-scale fluctuations
- **Why**: Robust to outliers, good for non-linear effects

### 5. Neural Network (MLP Regressor)
**Directory**: `neural_network/`
- **Purpose**: Capture complex non-linear interactions
- **Best for**: Both targets, especially returns
- **Why**: Flexible, handles complex feature interactions

### 6. ARIMA / SARIMAX
**Directory**: `arima/`
- **Purpose**: Classical time series forecasting
- **Best for**: Future close price levels
- **Why**: Captures autoregressive and moving average patterns

### 7. Ensemble / Hybrid
**Directory**: `ensemble/`
- **Purpose**: Combine multiple models for robust predictions
- **Best for**: Both targets with reduced overfitting risk
- **Why**: Combines strengths of different model types

## Quick Start

1. **Train all models**:
```bash
cd models/linear_regression && python train.py
cd ../random_forest && python train.py
cd ../xgboost && python train.py
cd ../svr && python train.py
cd ../neural_network && python train.py
cd ../arima && python train.py
cd ../ensemble && python train.py
```

2. **Make predictions**:
```python
from models.linear_regression.predict import predict as lr_predict
from models.utils import load_ml_data

data = load_ml_data()
X_new = data['X_test'].iloc[:10]

predictions = lr_predict(X_new, target='close')
```

3. **Evaluate models**:
```bash
cd models/linear_regression && python evaluate.py
```

## Dependencies

- `scikit-learn`: All models
- `xgboost`: XGBoost model (install with `pip install xgboost`)
- `statsmodels`: ARIMA model (install with `pip install statsmodels`)
- `pandas`, `numpy`: Data handling

## Model Selection Guide

| Target | Recommended Models | Why |
|--------|-------------------|-----|
| `future_close` | Linear Regression, ARIMA, Ensemble | Strong linear persistence |
| `future_returns` | Random Forest, XGBoost, Neural Network, Ensemble | Non-linear patterns, weak correlations |

## Notes

- All models use the same train/test split from `ml_data/` for consistency
- Models are saved in their respective directories as `.pkl` files
- Feature scaling is handled automatically by each model
- Ensemble model combines predictions from multiple base models
