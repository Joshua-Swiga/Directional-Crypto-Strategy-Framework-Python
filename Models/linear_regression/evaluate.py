"""
Evaluate Linear Regression models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Models.utils import load_ml_data, load_model, evaluate_predictions
import pandas as pd

def evaluate():
    """Evaluate saved Linear Regression models."""
    print("=" * 80)
    print("EVALUATING LINEAR REGRESSION MODELS")
    print("=" * 80)
    
    # Load data
    data = load_ml_data()
    X_test = data['X_test']
    y_close_test = data['y_close_test']
    y_returns_test = data['y_returns_test']
    
    model_dir = os.path.dirname(__file__)
    
    # Evaluate future_close model
    print("\n--- Evaluating Future Close Model ---")
    model_close, scaler_close = load_model(model_dir, 'linear_regression_close')
    # Model wrapper scales internally; pass raw features.
    y_pred_close = model_close.predict(X_test)
    metrics_close = evaluate_predictions(y_close_test, y_pred_close, "Future Close")
    
    # Evaluate future_returns model
    print("\n--- Evaluating Future Returns Model ---")
    model_returns, scaler_returns = load_model(model_dir, 'linear_regression_returns')
    y_pred_returns = model_returns.predict(X_test)
    metrics_returns = evaluate_predictions(y_returns_test, y_pred_returns, "Future Returns")
    
    return {
        'close': metrics_close,
        'returns': metrics_returns
    }

if __name__ == "__main__":
    evaluate()
