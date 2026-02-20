"""
Predict using Neural Network models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Models.utils import load_model
import pandas as pd
import numpy as np

def predict(X, target='close', model_dir=None):
    """Make predictions using trained Neural Network model."""
    if model_dir is None:
        model_dir = os.path.dirname(__file__)
    
    model_name = f'neural_network_{target}'
    model, scaler = load_model(model_dir, model_name)
    
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    
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
