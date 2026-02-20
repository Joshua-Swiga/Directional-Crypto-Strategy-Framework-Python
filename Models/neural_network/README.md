# Neural Network (MLP Regressor) Model

## Purpose
Capture complex non-linear interactions between features.

## Justification

- **Small network**: 7 features → small network enough, won't overfit easily.
- **Flexible**: Can model both level (future_close) and returns simultaneously with separate outputs if needed, or use independent networks.
- **Regularization**: Dropout or L2 regularization can prevent overfitting.

## Relevance

- **Feature engineering**: Useful if you want flexibility for engineered features (lags, rolling stats, interactions).
- **Return prediction**: Neural networks excel at return prediction, which is harder for linear models.
- **Complex patterns**: Can capture subtle non-linear relationships that tree-based models might miss.

## Architecture

- Hidden layers: 2-3 layers
- Neurons per layer: 64-128 (adjustable)
- Activation: ReLU
- Regularization: L2 (alpha parameter)
- Solver: 'adam' (adaptive learning rate)

## Hyperparameters

- `hidden_layer_sizes`: Tuple of layer sizes (default: (128, 64))
- `alpha`: L2 regularization (default: 0.001)
- `learning_rate`: Learning rate schedule (default: 'adaptive')
- `max_iter`: Maximum iterations (default: 500)

## Usage

```python
from models.neural_network.train import train_model
from models.neural_network.predict import predict

# Train
model_close, model_returns = train_model()

# Predict
predictions_close = predict(model_close, X_new, target='close')
predictions_returns = predict(model_returns, X_new, target='returns')
```
