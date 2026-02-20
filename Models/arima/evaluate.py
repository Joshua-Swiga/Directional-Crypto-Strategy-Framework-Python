"""
Evaluate ARIMA models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Models.utils import load_ml_data, evaluate_predictions
from Models.arima.model import ARIMAModel
import pandas as pd
import numpy as np
import joblib

def evaluate():
    """Evaluate saved ARIMA model."""
    print("=" * 80)
    print("EVALUATING ARIMA MODEL")
    print("=" * 80)
    
    # Load model
    model_dir = os.path.dirname(__file__)
    model = joblib.load(os.path.join(model_dir, 'arima_close.pkl'))
    
    # Load test data
    from env import smoothed_data
    df_full = pd.read_csv(smoothed_data)
    df_full['close'] = pd.to_numeric(df_full['close'], errors='coerce')
    y = df_full['close'].dropna()
    
    split_idx = int(len(y) * 0.8)
    y_test = y.iloc[split_idx:].values
    
    # Make predictions (simplified - in practice use proper walk-forward)
    predictions = model.predict(n_periods=len(y_test))
    
    metrics = evaluate_predictions(y_test, predictions, "ARIMA - Future Close")
    
    return metrics

if __name__ == "__main__":
    evaluate()
