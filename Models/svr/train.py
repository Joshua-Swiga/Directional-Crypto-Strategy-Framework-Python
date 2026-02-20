"""
Train SVR models for future_close and future_returns
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Models.utils import load_ml_data, save_model, evaluate_predictions
from Models.svr.model import SVRModel

def train_model(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'):
    """Train SVR models for both targets."""
    print("=" * 80)
    print("TRAINING SVR MODELS")
    print("=" * 80)
    
    data = load_ml_data()
    X_train = data['X_train']
    X_test = data['X_test']
    y_close_train = data['y_close_train']
    y_close_test = data['y_close_test']
    y_returns_train = data['y_returns_train']
    y_returns_test = data['y_returns_test']
    
    print("\n--- Training model for FUTURE CLOSE ---")
    model_close = SVRModel(target='close', kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
    model_close.fit(X_train, y_close_train)
    
    y_pred_train_close = model_close.predict(X_train)
    train_metrics_close = evaluate_predictions(y_close_train, y_pred_train_close, "Train - Future Close")
    
    y_pred_test_close = model_close.predict(X_test)
    test_metrics_close = evaluate_predictions(y_close_test, y_pred_test_close, "Test - Future Close")
    
    print("\n--- Training model for FUTURE RETURNS ---")
    model_returns = SVRModel(target='returns', kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
    model_returns.fit(X_train, y_returns_train)
    
    y_pred_train_returns = model_returns.predict(X_train)
    train_metrics_returns = evaluate_predictions(y_returns_train, y_pred_train_returns, "Train - Future Returns")
    
    y_pred_test_returns = model_returns.predict(X_test)
    test_metrics_returns = evaluate_predictions(y_returns_test, y_pred_test_returns, "Test - Future Returns")
    
    model_dir = os.path.dirname(__file__)
    save_model(model_close, model_close.scaler, model_dir, 'svr_close')
    save_model(model_returns, model_returns.scaler, model_dir, 'svr_returns')
    
    return model_close, model_returns

if __name__ == "__main__":
    train_model()
