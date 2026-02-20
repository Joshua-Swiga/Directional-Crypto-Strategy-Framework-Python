"""
Evaluate Ensemble models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Models.utils import load_ml_data, evaluate_predictions
import joblib

def evaluate():
    """Evaluate saved Ensemble models."""
    print("=" * 80)
    print("EVALUATING ENSEMBLE MODELS")
    print("=" * 80)
    
    data = load_ml_data()
    X_test = data['X_test']
    y_close_test = data['y_close_test']
    y_returns_test = data['y_returns_test']
    
    model_dir = os.path.dirname(__file__)
    
    print("\n--- Evaluating Future Close Ensemble ---")
    ensemble_close = joblib.load(os.path.join(model_dir, 'ensemble_close.pkl'))
    y_pred_close = ensemble_close.predict(X_test)
    metrics_close = evaluate_predictions(y_close_test, y_pred_close, "Ensemble Future Close")
    
    print("\n--- Evaluating Future Returns Ensemble ---")
    ensemble_returns = joblib.load(os.path.join(model_dir, 'ensemble_returns.pkl'))
    y_pred_returns = ensemble_returns.predict(X_test)
    metrics_returns = evaluate_predictions(y_returns_test, y_pred_returns, "Ensemble Future Returns")
    
    return {'close': metrics_close, 'returns': metrics_returns}

if __name__ == "__main__":
    evaluate()
