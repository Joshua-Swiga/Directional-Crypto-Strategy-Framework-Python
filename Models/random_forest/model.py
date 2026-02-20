"""
Random Forest Regressor Model
Handles non-linear relationships and feature interactions
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

class RandomForestModel:
    """Random Forest model wrapper."""
    
    def __init__(self, target='close', n_estimators=100, max_depth=10, 
                 min_samples_split=2, min_samples_leaf=1, random_state=42):
        """
        Initialize Random Forest model.
        
        Args:
            target: 'close' for future_close prediction, 'returns' for future_returns prediction
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples in leaf
            random_state: Random seed
        """
        self.target = target
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
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
    
    def get_feature_importance(self):
        """Get feature importances."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return {
            'importances': self.model.feature_importances_,
            'feature_names': getattr(self, 'feature_names', None)
        }
