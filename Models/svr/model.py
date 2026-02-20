"""
Support Vector Regression (SVR) Model
Non-linear regression with kernel trick
"""
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import numpy as np

class SVRModel:
    """SVR model wrapper."""
    
    def __init__(self, target='close', kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'):
        """
        Initialize SVR model.
        
        Args:
            target: 'close' for future_close prediction, 'returns' for future_returns prediction
            kernel: Kernel type ('rbf', 'linear', 'poly')
            C: Regularization parameter
            epsilon: Epsilon-tube width
            gamma: Kernel coefficient for 'rbf'
        """
        self.target = target
        self.model = SVR(
            kernel=kernel,
            C=C,
            epsilon=epsilon,
            gamma=gamma
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
