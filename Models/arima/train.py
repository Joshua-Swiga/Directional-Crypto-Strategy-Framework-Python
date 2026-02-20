"""
Train ARIMA/SARIMAX models for future_close
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Models.utils import load_ml_data, save_model, evaluate_predictions
from Models.arima.model import ARIMAModel
import pandas as pd
import numpy as np
import joblib

def train_model(order=(2, 1, 2), seasonal_order=None, use_exog=False):
    """
    Train ARIMA model for future_close.
    
    Args:
        order: (p, d, q) ARIMA order
        seasonal_order: (P, D, Q, s) Seasonal order (optional)
        use_exog: Whether to use exogenous features
    """
    print("=" * 80)
    print("TRAINING ARIMA/SARIMAX MODEL")
    print("=" * 80)
    
    data = load_ml_data()
    
    # For ARIMA, we need the time series of close prices
    # Load full dataset to get close prices
    from env import smoothed_data
    df_full = pd.read_csv(smoothed_data)
    df_full['close'] = pd.to_numeric(df_full['close'], errors='coerce')
    
    # Use close prices as the time series
    y = df_full['close'].dropna()
    
    # Split chronologically (same as ML models)
    split_idx = int(len(y) * 0.8)
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"\nTime series length: {len(y)}")
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Prepare exogenous features if needed
    exog_train = None
    exog_test = None
    if use_exog:
        # Use selected features as exogenous variables
        X_train = data['X_train']
        X_test = data['X_test']
        # Align indices
        exog_train = X_train.values[:len(y_train)]
        exog_test = X_test.values[:len(y_test)]
        print(f"Using {X_train.shape[1]} exogenous features")
    
    # Train model
    print("\n--- Training ARIMA model ---")
    model = ARIMAModel(target='close', order=order, seasonal_order=seasonal_order, use_exog=use_exog)
    
    if use_exog:
        exog_train_df = pd.DataFrame(exog_train) if exog_train is not None else None
        model.fit(y_train, exog=exog_train_df)
    else:
        model.fit(y_train)
    
    # Make predictions on test set (one-step ahead)
    print("\n--- Making predictions ---")
    predictions = []
    y_test_values = y_test.values
    
    # One-step ahead forecasting
    for i in range(len(y_test)):
        if use_exog and exog_test is not None:
            pred = model.predict(n_periods=1, exog=exog_test[i:i+1])
        else:
            pred = model.predict(n_periods=1)
        predictions.append(pred[0])
        
        # Update model with actual value (for next prediction)
        # In practice, you might retrain or use a rolling window
    
    predictions = np.array(predictions)
    
    # Evaluate
    test_metrics = evaluate_predictions(y_test_values, predictions, "ARIMA - Future Close")
    
    # Save model
    model_dir = os.path.dirname(__file__)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'arima_close.pkl'))
    print(f"✓ Model saved to {model_dir}")
    
    # Print summary
    print("\n--- Model Summary ---")
    print(model.get_summary())
    
    return model

if __name__ == "__main__":
    train_model(use_exog=False)  # Start with univariate ARIMA
