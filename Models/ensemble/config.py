"""
Ensemble configuration
Specify which models to include and their weights
"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# On Windows, the repo folder is `Models/` (capital M). Make this robust.
MODELS_FOLDER = 'Models' if os.path.exists(os.path.join(BASE_DIR, 'Models')) else 'models'

# Model directories
MODEL_DIRS = {
    'linear_regression': os.path.join(BASE_DIR, MODELS_FOLDER, 'linear_regression'),
    'random_forest': os.path.join(BASE_DIR, MODELS_FOLDER, 'random_forest'),
    'xgboost': os.path.join(BASE_DIR, MODELS_FOLDER, 'xgboost'),
    'svr': os.path.join(BASE_DIR, MODELS_FOLDER, 'svr'),
    'neural_network': os.path.join(BASE_DIR, MODELS_FOLDER, 'neural_network'),
    'arima': os.path.join(BASE_DIR, MODELS_FOLDER, 'arima')
}

# Ensemble weights for future_close prediction
# Weights should sum to 1.0
WEIGHTS_CLOSE = {
    'linear_regression': 0.3,  # Strong for level prediction
    'random_forest': 0.2,
    'xgboost': 0.2,
    'svr': 0.1,
    'neural_network': 0.15,
    'arima': 0.05  # Time series component
}

# Ensemble weights for future_returns prediction
WEIGHTS_RETURNS = {
    'linear_regression': 0.15,
    'random_forest': 0.25,  # Strong for non-linear returns
    'xgboost': 0.3,  # Strong for returns
    'svr': 0.15,
    'neural_network': 0.15,
    'arima': 0.0  # ARIMA less suitable for returns
}

# Models to include in ensemble
INCLUDE_MODELS = [
    'linear_regression',
    'random_forest',
    'xgboost',
    'svr',
    'neural_network'
    # 'arima'  # Uncomment if ARIMA is trained
]
