"""
Predict using Random Forest models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Models.utils import load_model
import pandas as pd
import numpy as np

def predict(X, target='close', model_dir=None):
    """
    Make predictions using trained Random Forest model.
    
    Args:
        X: Feature matrix (DataFrame or array)
        target: 'close' for future_close, 'returns' for future_returns
        model_dir: Directory containing saved model (default: current directory)
    
    Returns:
        Predictions array
    """
    if model_dir is None:
        model_dir = os.path.dirname(__file__)
    
    model_name = f'random_forest_{target}'
    model, scaler = load_model(model_dir, model_name)
    
    # Convert to DataFrame if needed
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    
    # Model wrapper scales internally; pass raw features.
    predictions = model.predict(X)
    
    return predictions

if __name__ == "__main__":
    from Models.utils import load_ml_data
    data = load_ml_data()
    X_test = data['X_test'].iloc[:10]
    
    pred_close = predict(X_test, target='close')
    pred_returns = predict(X_test, target='returns')
    
    print("Predictions (Future Close):", pred_close)
    print("Predictions (Future Returns):", pred_returns)
