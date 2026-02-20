"""
ARIMA / SARIMAX Model
Time series forecasting with optional exogenous features
"""
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. Install with: pip install statsmodels")

import numpy as np
import pandas as pd

class ARIMAModel:
    """ARIMA model wrapper."""
    
    def __init__(self, target='close', order=(2, 1, 2), seasonal_order=None, use_exog=False):
        """
        Initialize ARIMA model.
        
        Args:
            target: 'close' for future_close prediction
            order: (p, d, q) ARIMA order
            seasonal_order: (P, D, Q, s) Seasonal order (optional)
            use_exog: Whether to use exogenous features (SARIMAX)
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels not installed. Install with: pip install statsmodels")
        
        self.target = target
        self.order = order
        self.seasonal_order = seasonal_order
        self.use_exog = use_exog
        self.model = None
        self.is_fitted = False
        self.exog_features = None
    
    def fit(self, y, exog=None):
        """
        Fit the model.
        
        Args:
            y: Time series data (1D array or Series)
            exog: Exogenous features (optional, for SARIMAX)
        """
        if self.use_exog and exog is not None:
            # Use SARIMAX with exogenous variables
            self.exog_features = exog.columns.tolist() if isinstance(exog, pd.DataFrame) else None
            if self.seasonal_order:
                self.model = SARIMAX(y, exog=exog, order=self.order, seasonal_order=self.seasonal_order)
            else:
                self.model = SARIMAX(y, exog=exog, order=self.order)
        else:
            # Use ARIMA (univariate)
            self.model = ARIMA(y, order=self.order)
        
        self.fitted_model = self.model.fit()
        self.is_fitted = True
        return self
    
    def predict(self, n_periods=1, exog=None):
        """
        Make predictions.
        
        Args:
            n_periods: Number of periods to forecast
            exog: Exogenous features for future periods (if using SARIMAX)
        
        Returns:
            Forecast array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.fitted_model.forecast(steps=n_periods, exog=exog)
    
    def get_summary(self):
        """Get model summary statistics."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.fitted_model.summary()
