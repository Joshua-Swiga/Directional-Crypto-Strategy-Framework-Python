"""
Predict using Ensemble models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import joblib
import pandas as pd
import numpy as np

def predict(X, target='close', model_dir=None):
    """
    Make predictions using trained Ensemble model.
    
    Args:
        X: Feature matrix
        target: 'close' for future_close, 'returns' for future_returns
        model_dir: Directory containing saved model
    
    Returns:
        Ensemble predictions
    """
    if model_dir is None:
        model_dir = os.path.dirname(__file__)
    
    model_name = f'ensemble_{target}'
    model_path = os.path.join(model_dir, f'{model_name}.pkl')
    ensemble = joblib.load(model_path)
    
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    
    predictions = ensemble.predict(X)
    
    return predictions

if __name__ == "__main__":
    from Models.utils import load_ml_data
    data = load_ml_data()
    X_test = data['X_test'].iloc[:10]
    
    pred_close = predict(X_test, target='close')
    pred_returns = predict(X_test, target='returns')
    
    print("Ensemble Predictions (Future Close):", pred_close)
    print("Ensemble Predictions (Future Returns):", pred_returns)
