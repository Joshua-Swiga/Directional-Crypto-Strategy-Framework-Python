"""
Predict using ARIMA models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import joblib
import numpy as np
import pandas as pd

def predict(n_periods=1, exog=None, model_dir=None):
    """
    Make predictions using trained ARIMA model.
    
    Args:
        n_periods: Number of periods to forecast
        exog: Exogenous features for future periods (if using SARIMAX)
        model_dir: Directory containing saved model
    
    Returns:
        Forecast array
    """
    if model_dir is None:
        model_dir = os.path.dirname(__file__)
    
    model_path = os.path.join(model_dir, 'arima_close.pkl')
    model = joblib.load(model_path)
    
    if isinstance(exog, np.ndarray):
        exog = pd.DataFrame(exog)
    
    predictions = model.predict(n_periods=n_periods, exog=exog)
    
    return predictions

if __name__ == "__main__":
    pred = predict(n_periods=5)
    print("Predictions:", pred)
