"""
Train Ensemble model
Combines multiple base models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Models.utils import load_ml_data, save_model, evaluate_predictions
from Models.ensemble.model import EnsembleModel
import joblib

def train_ensemble():
    """Train ensemble model (loads pre-trained base models)."""
    print("=" * 80)
    print("TRAINING ENSEMBLE MODEL")
    print("=" * 80)
    print("\nNote: Ensemble combines pre-trained base models.")
    print("Ensure all base models are trained first.\n")
    
    data = load_ml_data()
    X_train = data['X_train']
    X_test = data['X_test']
    y_close_train = data['y_close_train']
    y_close_test = data['y_close_test']
    y_returns_train = data['y_returns_train']
    y_returns_test = data['y_returns_test']
    
    # Train ensemble for future_close
    print("\n--- Training ensemble for FUTURE CLOSE ---")
    ensemble_close = EnsembleModel(target='close')
    ensemble_close.load_models()
    
    # Evaluate on train
    y_pred_train_close = ensemble_close.predict(X_train)
    train_metrics_close = evaluate_predictions(y_close_train, y_pred_train_close, "Train - Ensemble Future Close")
    
    # Evaluate on test
    y_pred_test_close = ensemble_close.predict(X_test)
    test_metrics_close = evaluate_predictions(y_close_test, y_pred_test_close, "Test - Ensemble Future Close")
    
    # Train ensemble for future_returns
    print("\n--- Training ensemble for FUTURE RETURNS ---")
    ensemble_returns = EnsembleModel(target='returns')
    ensemble_returns.load_models()
    
    # Evaluate on train
    y_pred_train_returns = ensemble_returns.predict(X_train)
    train_metrics_returns = evaluate_predictions(y_returns_train, y_pred_train_returns, "Train - Ensemble Future Returns")
    
    # Evaluate on test
    y_pred_test_returns = ensemble_returns.predict(X_test)
    test_metrics_returns = evaluate_predictions(y_returns_test, y_pred_test_returns, "Test - Ensemble Future Returns")
    
    # Save ensemble models
    model_dir = os.path.dirname(__file__)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(ensemble_close, os.path.join(model_dir, 'ensemble_close.pkl'))
    joblib.dump(ensemble_returns, os.path.join(model_dir, 'ensemble_returns.pkl'))
    print(f"\nOK: Ensemble models saved to {model_dir}")
    
    # Print model contributions
    print("\n--- Model Contributions (sample predictions) ---")
    contributions_close = ensemble_close.get_model_contributions(X_test.iloc[:5])
    for model_name, preds in contributions_close.items():
        print(f"  {model_name}: {preds[:3]}...")
    
    return ensemble_close, ensemble_returns

if __name__ == "__main__":
    train_ensemble()
