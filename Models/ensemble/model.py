"""
Ensemble Model
Combines multiple models for robust predictions
"""
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Models.ensemble.config import WEIGHTS_CLOSE, WEIGHTS_RETURNS, INCLUDE_MODELS, MODEL_DIRS

class EnsembleModel:
    """Ensemble model that combines multiple base models."""
    
    def __init__(self, target='close', weights=None, include_models=None):
        """
        Initialize Ensemble model.
        
        Args:
            target: 'close' for future_close, 'returns' for future_returns
            weights: Dictionary of model weights (if None, uses config)
            include_models: List of model names to include (if None, uses config)
        """
        self.target = target
        self.weights = weights or (WEIGHTS_CLOSE if target == 'close' else WEIGHTS_RETURNS)
        self.include_models = include_models or INCLUDE_MODELS
        self.models = {}
        self.is_fitted = False
    
    def load_models(self):
        """Load all base models."""
        for model_name in self.include_models:
            try:
                if model_name == 'linear_regression':
                    from Models.linear_regression.predict import predict as lr_predict
                    def make_predictor(mn, pred_func):
                        return lambda X: pred_func(X, target=self.target, model_dir=MODEL_DIRS[mn])
                    self.models[model_name] = make_predictor(model_name, lr_predict)
                elif model_name == 'random_forest':
                    from Models.random_forest.predict import predict as rf_predict
                    def make_predictor(mn, pred_func):
                        return lambda X: pred_func(X, target=self.target, model_dir=MODEL_DIRS[mn])
                    self.models[model_name] = make_predictor(model_name, rf_predict)
                elif model_name == 'xgboost':
                    from Models.xgboost.predict import predict as xgb_predict
                    def make_predictor(mn, pred_func):
                        return lambda X: pred_func(X, target=self.target, model_dir=MODEL_DIRS[mn])
                    self.models[model_name] = make_predictor(model_name, xgb_predict)
                elif model_name == 'svr':
                    from Models.svr.predict import predict as svr_predict
                    def make_predictor(mn, pred_func):
                        return lambda X: pred_func(X, target=self.target, model_dir=MODEL_DIRS[mn])
                    self.models[model_name] = make_predictor(model_name, svr_predict)
                elif model_name == 'neural_network':
                    from Models.neural_network.predict import predict as nn_predict
                    def make_predictor(mn, pred_func):
                        return lambda X: pred_func(X, target=self.target, model_dir=MODEL_DIRS[mn])
                    self.models[model_name] = make_predictor(model_name, nn_predict)
                elif model_name == 'arima':
                    # ARIMA requires different handling
                    print(f"Warning: ARIMA not yet integrated into ensemble")
                    continue
            except Exception as e:
                print(f"Warning: Could not load {model_name}: {e}")
                continue
        
        self.is_fitted = True
        return self

    def __getstate__(self):
        """
        Make the model pickle-safe.

        We intentionally do NOT pickle runtime predictor callables (lambdas)
        stored in `self.models`. They are reconstructed on demand by `load_models()`.
        """
        state = dict(self.__dict__)
        state["models"] = {}
        state["is_fitted"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.models = {}
    
    def predict(self, X):
        """
        Make ensemble predictions.
        
        Args:
            X: Feature matrix
        
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            self.load_models()
        
        predictions = {}
        for model_name, predict_func in self.models.items():
            try:
                pred = predict_func(X)
                predictions[model_name] = pred
            except Exception as e:
                print(f"Warning: {model_name} prediction failed: {e}")
                continue
        
        # Weighted average
        if not predictions:
            raise ValueError("No models available for prediction")
        
        # Normalize weights to sum to 1
        available_models = list(predictions.keys())
        total_weight = sum(self.weights.get(m, 0) for m in available_models)
        if total_weight == 0:
            # Equal weights if no weights specified
            weights = {m: 1.0/len(available_models) for m in available_models}
        else:
            weights = {m: self.weights.get(m, 0) / total_weight for m in available_models}
        
        # Compute weighted average
        ensemble_pred = np.zeros(len(X))
        for model_name, pred in predictions.items():
            ensemble_pred += weights[model_name] * pred
        
        return ensemble_pred
    
    def get_model_contributions(self, X):
        """Get individual model predictions for analysis."""
        if not self.is_fitted:
            self.load_models()
        
        contributions = {}
        for model_name, predict_func in self.models.items():
            try:
                pred = predict_func(X)
                contributions[model_name] = pred
            except Exception as e:
                print(f"Warning: {model_name} prediction failed: {e}")
        
        return contributions
