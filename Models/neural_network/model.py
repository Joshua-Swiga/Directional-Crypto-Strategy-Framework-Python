"""
Neural Network (MLP Regressor) Model
Multi-layer perceptron for complex non-linear relationships
"""
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

class NeuralNetworkModel:
    """Neural Network model wrapper."""
    
    def __init__(self, target='close', hidden_layer_sizes=(128, 64), alpha=0.001,
                 learning_rate='adaptive', max_iter=500, random_state=42):
        """
        Initialize Neural Network model.
        
        Args:
            target: 'close' for future_close prediction, 'returns' for future_returns prediction
            hidden_layer_sizes: Tuple of layer sizes
            alpha: L2 regularization parameter
            learning_rate: Learning rate schedule ('constant', 'invscaling', 'adaptive')
            max_iter: Maximum iterations
            random_state: Random seed
        """
        self.target = target
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X, y):
        """Fit the model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
