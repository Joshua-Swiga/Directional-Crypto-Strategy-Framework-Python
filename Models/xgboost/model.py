"""
XGBoost Regressor Model
Gradient boosting for high-accuracy regression
"""
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

from sklearn.preprocessing import StandardScaler
import numpy as np

class XGBoostModel:
    """XGBoost model wrapper."""
    
    def __init__(self, target='close', n_estimators=100, max_depth=6, 
                 learning_rate=0.1, subsample=0.8, random_state=42):
        """
        Initialize XGBoost model.
        
        Args:
            target: 'close' for future_close prediction, 'returns' for future_returns prediction
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Step size shrinkage
            subsample: Fraction of samples for training
            random_state: Random seed
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        self.target = target
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
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
