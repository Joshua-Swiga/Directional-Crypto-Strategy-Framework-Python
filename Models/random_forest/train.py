"""
Train Random Forest models for future_close and future_returns
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Models.utils import load_ml_data, save_model, evaluate_predictions
from Models.random_forest.model import RandomForestModel
import pandas as pd

def train_model(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1):
    """Train Random Forest models for both targets."""
    print("=" * 80)
    print("TRAINING RANDOM FOREST MODELS")
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
    model_close = RandomForestModel(
        target='close',
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
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
    model_returns = RandomForestModel(
        target='returns',
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
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
    save_model(model_close, model_close.scaler, model_dir, 'random_forest_close')
    save_model(model_returns, model_returns.scaler, model_dir, 'random_forest_returns')
    
    # Print feature importances
    print("\n--- Feature Importances (Future Close) ---")
    imp_close = model_close.get_feature_importance()
    if imp_close['feature_names']:
        for name, imp in sorted(zip(imp_close['feature_names'], imp_close['importances']), 
                               key=lambda x: x[1], reverse=True):
            print(f"  {name}: {imp:.6f}")
    
    print("\n--- Feature Importances (Future Returns) ---")
    imp_returns = model_returns.get_feature_importance()
    if imp_returns['feature_names']:
        for name, imp in sorted(zip(imp_returns['feature_names'], imp_returns['importances']), 
                               key=lambda x: x[1], reverse=True):
            print(f"  {name}: {imp:.6f}")
    
    return model_close, model_returns

if __name__ == "__main__":
    train_model()
