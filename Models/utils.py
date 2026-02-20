"""
Common utilities for all models
"""
import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DATA_DIR = os.path.join(BASE_DIR, 'ml_data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def load_ml_data():
    """Load ML-ready data from ml_data directory."""
    X_train = pd.read_csv(os.path.join(ML_DATA_DIR, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(ML_DATA_DIR, 'X_test.csv'))
    # pandas removed `squeeze=`; read as 1-col DataFrame then take the first column.
    y_close_train = pd.read_csv(os.path.join(ML_DATA_DIR, 'y_close_train.csv')).iloc[:, 0]
    y_close_test = pd.read_csv(os.path.join(ML_DATA_DIR, 'y_close_test.csv')).iloc[:, 0]
    y_returns_train = pd.read_csv(os.path.join(ML_DATA_DIR, 'y_returns_train.csv')).iloc[:, 0]
    y_returns_test = pd.read_csv(os.path.join(ML_DATA_DIR, 'y_returns_test.csv')).iloc[:, 0]
    
    # Load feature metadata
    with open(os.path.join(ML_DATA_DIR, 'feature_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_close_train': y_close_train,
        'y_close_test': y_close_test,
        'y_returns_train': y_returns_train,
        'y_returns_test': y_returns_test,
        'metadata': metadata
    }

def get_scaler():
    """Get fitted scaler (assumes StandardScaler was used)."""
    scaler = StandardScaler()
    data = load_ml_data()
    scaler.fit(data['X_train'])
    return scaler

def evaluate_predictions(y_true, y_pred, model_name="Model"):
    """Evaluate predictions and return metrics."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate mean absolute percentage error
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }
    
    print(f"\n{model_name} Metrics:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R2:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return metrics

def save_model(model, scaler, model_dir, model_name):
    """Save model and scaler to model directory."""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, f'{model_name}.pkl'))
    if scaler is not None:
        joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    print(f"OK: Model saved to {model_dir}")

def load_model(model_dir, model_name):
    """Load model and scaler from model directory."""
    model = joblib.load(os.path.join(model_dir, f'{model_name}.pkl'))
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler
