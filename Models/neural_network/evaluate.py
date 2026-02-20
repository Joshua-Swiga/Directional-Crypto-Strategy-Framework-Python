"""
Evaluate Neural Network models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Models.utils import load_ml_data, load_model, evaluate_predictions

def evaluate():
    """Evaluate saved Neural Network models."""
    print("=" * 80)
    print("EVALUATING NEURAL NETWORK MODELS")
    print("=" * 80)
    
    data = load_ml_data()
    X_test = data['X_test']
    y_close_test = data['y_close_test']
    y_returns_test = data['y_returns_test']
    
    model_dir = os.path.dirname(__file__)
    
    print("\n--- Evaluating Future Close Model ---")
    model_close, scaler_close = load_model(model_dir, 'neural_network_close')
    y_pred_close = model_close.predict(X_test)
    metrics_close = evaluate_predictions(y_close_test, y_pred_close, "Future Close")
    
    print("\n--- Evaluating Future Returns Model ---")
    model_returns, scaler_returns = load_model(model_dir, 'neural_network_returns')
    y_pred_returns = model_returns.predict(X_test)
    metrics_returns = evaluate_predictions(y_returns_test, y_pred_returns, "Future Returns")
    
    return {'close': metrics_close, 'returns': metrics_returns}

if __name__ == "__main__":
    evaluate()
