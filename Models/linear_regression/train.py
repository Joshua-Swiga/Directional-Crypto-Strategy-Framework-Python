"""
Train Linear Regression models for future_close and future_returns
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Models.utils import load_ml_data, save_model, evaluate_predictions
from Models.linear_regression.model import LinearRegressionModel
import pandas as pd

def train_model():
    """Train Linear Regression models for both targets."""
    print("=" * 80)
    print("TRAINING LINEAR REGRESSION MODELS")
    print("=" * 80)
    
    # Load data
    data = load_ml_data()
    X_train = data['X_train']
    X_test = data['X_test']
    y_close_train = data['y_close_train']
    y_close_test = data['y_close_test']
    y_returns_train = data['y_returns_train']
    y_returns_test = data['y_returns_test']
    
    # Train model for future_close
    print("\n--- Training model for FUTURE CLOSE ---")
    model_close = LinearRegressionModel(target='close')
    model_close.feature_names = X_train.columns.tolist()
    model_close.fit(X_train, y_close_train)
    
    # Evaluate on train
    y_pred_train_close = model_close.predict(X_train)
    train_metrics_close = evaluate_predictions(y_close_train, y_pred_train_close, "Train - Future Close")
    
    # Evaluate on test
    y_pred_test_close = model_close.predict(X_test)
    test_metrics_close = evaluate_predictions(y_close_test, y_pred_test_close, "Test - Future Close")
    
    # Train model for future_returns
    print("\n--- Training model for FUTURE RETURNS ---")
    model_returns = LinearRegressionModel(target='returns')
    model_returns.feature_names = X_train.columns.tolist()
    model_returns.fit(X_train, y_returns_train)
    
    # Evaluate on train
    y_pred_train_returns = model_returns.predict(X_train)
    train_metrics_returns = evaluate_predictions(y_returns_train, y_pred_train_returns, "Train - Future Returns")
    
    # Evaluate on test
    y_pred_test_returns = model_returns.predict(X_test)
    test_metrics_returns = evaluate_predictions(y_returns_test, y_pred_test_returns, "Test - Future Returns")
    
    # Save models
    model_dir = os.path.dirname(__file__)
    save_model(model_close, model_close.scaler, model_dir, 'linear_regression_close')
    save_model(model_returns, model_returns.scaler, model_dir, 'linear_regression_returns')
    
    # Print coefficients
    print("\n--- Feature Coefficients (Future Close) ---")
    coef_close = model_close.get_coefficients()
    for i, (name, coef) in enumerate(zip(coef_close['feature_names'], coef_close['coefficients'])):
        print(f"  {name}: {coef:.6f}")
    print(f"  Intercept: {coef_close['intercept']:.6f}")
    
    print("\n--- Feature Coefficients (Future Returns) ---")
    coef_returns = model_returns.get_coefficients()
    for i, (name, coef) in enumerate(zip(coef_returns['feature_names'], coef_returns['coefficients'])):
        print(f"  {name}: {coef:.6f}")
    print(f"  Intercept: {coef_returns['intercept']:.6f}")
    
    return model_close, model_returns

if __name__ == "__main__":
    train_model()
