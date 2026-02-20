"""
Linear Regression Model
Baseline model for predicting future_close and future_returns
"""
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

class LinearRegressionModel:
    """Linear Regression model wrapper."""
    
    def __init__(self, target='close'):
        """
        Initialize Linear Regression model.
        
        Args:
            target: 'close' for future_close prediction, 'returns' for future_returns prediction
        """
        self.target = target
        self.model = LinearRegression()
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
    
    def get_coefficients(self):
        """Get feature coefficients."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return {
            'coefficients': self.model.coef_,
            'intercept': self.model.intercept_,
            'feature_names': getattr(self, 'feature_names', None)
        }
