"""
Exploratory Analysis & Correlation - Strategy Feature Set
Covers all derived and raw numeric features; full correlation analysis and strategy-relevant summary.
Includes automatic redundant feature removal, strong predictor selection, and out-of-sample validation.
"""
import pandas as pd
import numpy as np
from env import smoothed_data
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------------------------
# 1. LOAD DATA & DEFINE ALL FEATURES
# ---------------------------------------------------------------------------
df = pd.read_csv(smoothed_data)

# All numeric feature columns we want to analyze (exclude time identifiers)
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'quote_volume', 'num_trades',
    'close_sma_10', 'candle_body', 'candle_range', 'volatility',
    'bb_upper', 'bb_middle', 'bb_lower', 'garman_klass', 'rsi', 'atr'
]
# Use only columns that exist
features = [c for c in FEATURE_COLUMNS if c in df.columns]
if not features:
    raise ValueError("No feature columns found. Ensure smoothed_data includes derived features.")

df_features = df[features].copy()

# Ensure numeric types (coerce errors to NaN)
for col in features:
    df_features[col] = pd.to_numeric(df_features[col], errors='coerce')

# Drop rows with NaN so correlation and targets are valid
df_clean = df_features.dropna(how='any')
n_dropped = len(df_features) - len(df_clean)
if n_dropped > 0:
    print(f"Dropped {n_dropped} rows with NaN in features (e.g. rolling warm-up).\n")

# Reattach clean index to original df for target alignment
df = df.loc[df_clean.index].copy()
df_clean = df_clean.copy()
for col in features:
    df[col] = df_clean[col].values

# Create target variables
df['future_close'] = df['close'].shift(-1)
df['returns'] = df['close'].pct_change()
df['future_returns'] = df['future_close'].pct_change()

# Align: remove last row (no future_close)
df = df.iloc[:-1]
df_clean = df_clean.iloc[:-1]
for col in features:
    df[col] = df_clean[col].values

# ---------------------------------------------------------------------------
# 2. EXPLORATORY SUMMARY
# ---------------------------------------------------------------------------
print("=" * 80)
print("EXPLORATORY DATA SUMMARY")
print("=" * 80)

print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Feature set: {len(features)} numeric features")
print(f"Features: {features}")

print("\n--- Missing values (after dropna on features) ---")
missing = df[features].isna().sum()
print(missing[missing > 0] if missing.any() else "None")

# print("\n--- Basic statistics (features) ---")
# print(df[features].describe().round(4).to_string())

# Strategy-relevant univariate summaries
print("\n--- Strategy-relevant indicator ranges ---")
if 'rsi' in df.columns:
    print(f"  RSI: min={df['rsi'].min():.2f}, max={df['rsi'].max():.2f}, mean={df['rsi'].mean():.2f}")
if 'atr' in df.columns:
    print(f"  ATR: min={df['atr'].min():.6f}, max={df['atr'].max():.6f}, mean={df['atr'].mean():.6f}")
if 'garman_klass' in df.columns:
    print(f"  Garman-Klass: min={df['garman_klass'].min():.6f}, max={df['garman_klass'].max():.6f}")

# ---------------------------------------------------------------------------
# 3. CORRELATION MATRICES
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("CORRELATION MATRIX - ALL FEATURES")
print("=" * 80)

corr_matrix = df[features].corr()
# print(corr_matrix.round(4).to_string())

# Correlations with targets
print("\n" + "=" * 80)
print("CORRELATION WITH FUTURE CLOSE PRICE")
print("=" * 80)
corr_future_close = df[features].corrwith(df['future_close']).sort_values(ascending=False)
print(corr_future_close.round(4).to_string())

print("\n" + "=" * 80)
print("CORRELATION WITH FUTURE RETURNS")
print("=" * 80)
corr_future_returns = df[features].corrwith(df['future_returns']).sort_values(ascending=False)
print(corr_future_returns.round(4).to_string())

# ---------------------------------------------------------------------------
# 4. CORRELATION ASPECTS - REDUNDANCY, NEGATIVE, PREDICTORS, STRENGTH
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("CORRELATION ASPECTS - INTERPRETATION")
print("=" * 80)

# 4.1 Redundant features (high pairwise correlation)
print("\n1. REDUNDANT FEATURES (|correlation| > 0.85 between features)")
print("-" * 80)
high_corr_pairs = []
for i in range(len(features)):
    for j in range(i + 1, len(features)):
        c = corr_matrix.iloc[i, j]
        if abs(c) > 0.85:
            high_corr_pairs.append((features[i], features[j], c))
if high_corr_pairs:
    for f1, f2, c in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"  - {f1} <-> {f2}: {c:.4f}  -> consider dropping one to reduce multicollinearity")
else:
    print("  - No pairs with |r| > 0.85")

# 4.2 Negative correlations
print("\n2. NEGATIVE CORRELATIONS (|r| > 0.3 between features)")
print("-" * 80)
neg_pairs = []
for i in range(len(features)):
    for j in range(i + 1, len(features)):
        c = corr_matrix.iloc[i, j]
        if c < -0.3:
            neg_pairs.append((features[i], features[j], c))
if neg_pairs:
    for f1, f2, c in sorted(neg_pairs, key=lambda x: x[2]):
        print(f"  - {f1} <-> {f2}: {c:.4f}  -> inverse relationship / potential hedging or mean reversion")
else:
    print("  - No strong negative feature pairs (< -0.3)")

# 4.3 Candidate predictors (vs future close and vs future returns)
print("\n3. CANDIDATE PREDICTORS (|correlation with target| > 0.10)")
print("-" * 80)
threshold = 0.10
pred_close = corr_future_close[abs(corr_future_close) > threshold].sort_values(ascending=False)
pred_ret   = corr_future_returns[abs(corr_future_returns) > threshold].sort_values(ascending=False)
print("  vs Future Close:")
for f, v in pred_close.items():
    print(f"    - {f}: {v:.4f}")
print("  vs Future Returns:")
for f, v in pred_ret.items():
    print(f"    - {f}: {v:.4f}")
if pred_close.empty and pred_ret.empty:
    print("  (None above threshold; linear correlation with next-period targets is weak - consider lags or non-linear models)")

# 4.4 Feature strength summary (absolute correlation with future close)
print("\n4. FEATURE STRENGTH vs FUTURE CLOSE (for level prediction)")
print("-" * 80)
abs_close = corr_future_close.abs().sort_values(ascending=False)
for f in abs_close.index:
    v = abs_close[f]
    level = "STRONG" if v > 0.2 else "MODERATE" if v > 0.1 else "WEAK"
    print(f"  - {f}: {corr_future_close[f]:.4f}  (|r|={v:.4f}, {level})")

# 4.5 Feature strength vs future returns
print("\n5. FEATURE STRENGTH vs FUTURE RETURNS (for return prediction)")
print("-" * 80)
abs_ret = corr_future_returns.abs().sort_values(ascending=False)
for f in abs_ret.index:
    v = abs_ret[f]
    level = "STRONG" if v > 0.2 else "MODERATE" if v > 0.1 else "WEAK"
    print(f"  - {f}: {corr_future_returns[f]:.4f}  (|r|={v:.4f}, {level})")

# ---------------------------------------------------------------------------
# 5. STRATEGY-RELEVANT SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STRATEGY-RELEVANT SUMMARY")
print("=" * 80)
print("""
- Leading / useful for prediction:
  - Features with higher |correlation| to future_close or future_returns are candidate inputs for models.
  - RSI, ATR, Bollinger positions, and volatility (incl. Garman-Klass) often lead regime or momentum.

- Multicollinearity:
  - Pairs with |r| > 0.85 are redundant; use one or combine (e.g. keep one of open/high/low/close if all included).
  - BB middle is highly correlated with close_sma; consider using either trend or band position, not both raw.

- Correlation with targets:
  - Future close is usually highly correlated with current close/open/high/low (persistence).
  - Future returns are typically weakly correlated with levels; focus on returns, volatility, and momentum (RSI, ATR) for return prediction.

- Next steps:
  - Use this feature set in models; consider lagged features or rolling stats if needed.
  - Validate on out-of-sample or walk-forward to avoid overfitting.
""")

# ---------------------------------------------------------------------------
# 6. VISUALIZATION
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SAVING CORRELATION PLOTS")
print("=" * 80)

out_dir = r'C:\Users\JoshuaSwiga\Desktop\strat\correlation_images'
os.makedirs(out_dir, exist_ok=True)

# Full feature correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Heatmap 1: Feature x Feature
im1 = axes[0].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
axes[0].set_xticks(range(len(features)))
axes[0].set_yticks(range(len(features)))
axes[0].set_xticklabels(features, rotation=45, ha='right', fontsize=8)
axes[0].set_yticklabels(features, fontsize=8)
axes[0].set_title('Feature Correlation Matrix (all features)')
for i in range(len(features)):
    for j in range(len(features)):
        axes[0].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black', fontsize=6)

# Heatmap 2: Features x Targets
future_corr_df = pd.DataFrame({
    'Future Close': corr_future_close,
    'Future Returns': corr_future_returns
})
im2 = axes[1].imshow(future_corr_df, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
axes[1].set_xticks(range(2))
axes[1].set_yticks(range(len(features)))
axes[1].set_xticklabels(['Future Close', 'Future Returns'])
axes[1].set_yticklabels(features, fontsize=8)
axes[1].set_title('Correlations with Future Price & Returns')
for i in range(len(features)):
    for j in range(2):
        axes[1].text(j, i, f'{future_corr_df.iloc[i, j]:.2f}', ha='center', va='center', color='black', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'correlation_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()
print("OK: Saved correlation_analysis.png (feature matrix + target correlations)")

# ---------------------------------------------------------------------------
# 7. AUTOMATIC REDUNDANT FEATURE REMOVAL & COMBINATION
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("FEATURE SELECTION: REMOVING REDUNDANT FEATURES")
print("=" * 80)

def remove_redundant_features(df_features, corr_matrix, target_corr, threshold=0.85):
    """
    Remove redundant features by keeping the one with higher correlation to target.
    If features are highly correlated (>threshold), keep the one with stronger predictive power.
    """
    features_list = list(df_features.columns)
    to_remove = set()
    to_keep = set(features_list)
    
    redundant_pairs = []
    for i in range(len(features_list)):
        for j in range(i + 1, len(features_list)):
            f1, f2 = features_list[i], features_list[j]
            if f1 in to_remove or f2 in to_remove:
                continue
            corr_val = corr_matrix.loc[f1, f2]
            if abs(corr_val) > threshold:
                redundant_pairs.append((f1, f2, corr_val))
                # Keep the feature with higher absolute correlation to target
                corr1 = abs(target_corr.get(f1, 0))
                corr2 = abs(target_corr.get(f2, 0))
                if corr1 >= corr2:
                    to_remove.add(f2)
                    print(f"  Removing {f2} (redundant with {f1}, r={corr_val:.4f}, keeping {f1} with |target_r|={corr1:.4f})")
                else:
                    to_remove.add(f1)
                    print(f"  Removing {f1} (redundant with {f2}, r={corr_val:.4f}, keeping {f2} with |target_r|={corr2:.4f})")
    
    selected_features = [f for f in features_list if f not in to_remove]
    return selected_features, redundant_pairs

# Remove redundant features for both targets
print("\n--- Removing redundant features for FUTURE CLOSE prediction ---")
features_close, redundant_close = remove_redundant_features(
    df[features], corr_matrix, corr_future_close.to_dict(), threshold=0.85
)

print("\n--- Removing redundant features for FUTURE RETURNS prediction ---")
features_returns, redundant_returns = remove_redundant_features(
    df[features], corr_matrix, corr_future_returns.to_dict(), threshold=0.85
)

print(f"\nOriginal features: {len(features)}")
print(f"Selected for close prediction: {len(features_close)}")
print(f"Selected for returns prediction: {len(features_returns)}")

# ---------------------------------------------------------------------------
# 8. STRONG PREDICTOR SELECTION
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("FEATURE SELECTION: SELECTING STRONG PREDICTORS")
print("=" * 80)

def select_strong_predictors(feature_list, target_corr, min_corr=0.10):
    """Select features with |correlation| > min_corr to target."""
    strong = [f for f in feature_list if abs(target_corr.get(f, 0)) > min_corr]
    return sorted(strong, key=lambda x: abs(target_corr.get(x, 0)), reverse=True)

# Select strong predictors
strong_close = select_strong_predictors(features_close, corr_future_close.to_dict(), min_corr=0.10)
strong_returns = select_strong_predictors(features_returns, corr_future_returns.to_dict(), min_corr=0.10)

print(f"\nStrong predictors for FUTURE CLOSE (|r| > 0.10): {len(strong_close)}")
for f in strong_close:
    print(f"  - {f}: {corr_future_close[f]:.4f}")

print(f"\nStrong predictors for FUTURE RETURNS (|r| > 0.10): {len(strong_returns)}")
for f in strong_returns:
    print(f"  - {f}: {corr_future_returns[f]:.4f}")

# Use combined set for ML (union of strong predictors from both targets)
# IMPORTANT: keep deterministic ordering for stable training/prediction.
ml_features = sorted(set(strong_close + strong_returns))
if not ml_features:
    print("\nWARNING: No strong predictors found. Using all non-redundant features.")
    ml_features = sorted(set(features_close + features_returns))

print(f"\nFinal ML feature set (union): {len(ml_features)} features")
print(f"Features: {ml_features}")

# ---------------------------------------------------------------------------
# 9. OUT-OF-SAMPLE VALIDATION SETUP
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("OUT-OF-SAMPLE VALIDATION SETUP")
print("=" * 80)

# Prepare data for ML
X = df[ml_features].copy()
y_close = df['future_close'].copy()
y_returns = df['future_returns'].copy()

# Remove any remaining NaN
valid_idx = ~(X.isna().any(axis=1) | y_close.isna() | y_returns.isna())
X = X[valid_idx].copy()
y_close = y_close[valid_idx].copy()
y_returns = y_returns[valid_idx].copy()

print(f"\nData shape: {X.shape[0]} samples, {X.shape[1]} features")

# Train/test split (80/20) - chronological split (no shuffle for time series)
split_idx = int(len(X) * 0.8)
X_train = X.iloc[:split_idx].copy()
X_test = X.iloc[split_idx:].copy()
y_close_train = y_close.iloc[:split_idx].copy()
y_close_test = y_close.iloc[split_idx:].copy()
y_returns_train = y_returns.iloc[:split_idx].copy()
y_returns_test = y_returns.iloc[split_idx:].copy()

print(f"Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# Scale features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# ---------------------------------------------------------------------------
# 10. BASELINE MODEL VALIDATION
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("BASELINE MODEL VALIDATION (Linear Regression)")
print("=" * 80)

def evaluate_model(y_true, y_pred, target_name):
    """Evaluate model and return metrics."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{target_name} Prediction Metrics:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R2:   {r2:.4f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

# Model 1: Predict Future Close
print("\n--- Model 1: Future Close Price ---")
model_close = LinearRegression()
model_close.fit(X_train_scaled, y_close_train)
y_close_pred_train = model_close.predict(X_train_scaled)
y_close_pred_test = model_close.predict(X_test_scaled)

print("\nTRAIN SET:")
train_metrics_close = evaluate_model(y_close_train, y_close_pred_train, "Future Close")
print("\nTEST SET (OUT-OF-SAMPLE):")
test_metrics_close = evaluate_model(y_close_test, y_close_pred_test, "Future Close")

# Model 2: Predict Future Returns
print("\n--- Model 2: Future Returns ---")
model_returns = LinearRegression()
model_returns.fit(X_train_scaled, y_returns_train)
y_returns_pred_train = model_returns.predict(X_train_scaled)
y_returns_pred_test = model_returns.predict(X_test_scaled)

print("\nTRAIN SET:")
train_metrics_returns = evaluate_model(y_returns_train, y_returns_pred_train, "Future Returns")
print("\nTEST SET (OUT-OF-SAMPLE):")
test_metrics_returns = evaluate_model(y_returns_test, y_returns_pred_test, "Future Returns")

# Overfitting check
print("\n--- Overfitting Check ---")
close_overfit = test_metrics_close['r2'] < train_metrics_close['r2'] * 0.7
returns_overfit = test_metrics_returns['r2'] < train_metrics_returns['r2'] * 0.7

if close_overfit:
    print("WARNING: Future Close model may be overfitting (test R2 < 70% of train R2)")
else:
    print("OK: Future Close model generalization looks acceptable")

if returns_overfit:
    print("WARNING: Future Returns model may be overfitting (test R2 < 70% of train R2)")
else:
    print("OK: Future Returns model generalization looks acceptable")

# ---------------------------------------------------------------------------
# 11. SAVE CLEANED FEATURE SET FOR ML MODELS
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SAVING CLEANED FEATURE SET")
print("=" * 80)

# Save ML-ready dataset
ml_data_dir = r'C:\Users\JoshuaSwiga\Desktop\strat\ml_data'
os.makedirs(ml_data_dir, exist_ok=True)

# Save full dataset with selected features
ml_dataset = df[ml_features + ['future_close', 'future_returns']].copy()
ml_dataset = ml_dataset[valid_idx].copy()
ml_dataset.to_csv(os.path.join(ml_data_dir, 'ml_ready_features.csv'), index=False)
print(f"OK: Saved ML-ready dataset: ml_ready_features.csv ({len(ml_dataset)} rows)")

# Save train/test splits
X_train.to_csv(os.path.join(ml_data_dir, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(ml_data_dir, 'X_test.csv'), index=False)
y_close_train.to_csv(os.path.join(ml_data_dir, 'y_close_train.csv'), index=False)
y_close_test.to_csv(os.path.join(ml_data_dir, 'y_close_test.csv'), index=False)
y_returns_train.to_csv(os.path.join(ml_data_dir, 'y_returns_train.csv'), index=False)
y_returns_test.to_csv(os.path.join(ml_data_dir, 'y_returns_test.csv'), index=False)
print("OK: Saved train/test splits to ml_data/")

# Save feature list and metadata
feature_metadata = {
    'original_features': features,
    'ml_features': ml_features,
    'features_for_close': features_close,
    'features_for_returns': features_returns,
    'strong_predictors_close': strong_close,
    'strong_predictors_returns': strong_returns,
    'train_size': len(X_train),
    'test_size': len(X_test),
    'validation_metrics': {
        'close': {'train': train_metrics_close, 'test': test_metrics_close},
        'returns': {'train': train_metrics_returns, 'test': test_metrics_returns}
    }
}

import json
with open(os.path.join(ml_data_dir, 'feature_metadata.json'), 'w') as f:
    json.dump(feature_metadata, f, indent=2, default=str)
print("OK: Saved feature metadata: feature_metadata.json")

# Save summary report
summary_report = f"""
FEATURE SELECTION & VALIDATION SUMMARY
========================================

Original Features: {len(features)}
Selected ML Features: {len(ml_features)}

Redundant Features Removed:
  - Close prediction: {len(features) - len(features_close)} removed
  - Returns prediction: {len(features) - len(features_returns)} removed

Strong Predictors Selected:
  - Future Close: {len(strong_close)} features
  - Future Returns: {len(strong_returns)} features

Out-of-Sample Validation:
  - Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)
  - Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)

Model Performance (Linear Regression Baseline):
  Future Close:
    Train R2: {train_metrics_close['r2']:.4f}
    Test R2:  {test_metrics_close['r2']:.4f}
    Test RMSE: {test_metrics_close['rmse']:.6f}
  
  Future Returns:
    Train R2: {train_metrics_returns['r2']:.4f}
    Test R2:  {test_metrics_returns['r2']:.4f}
    Test RMSE: {test_metrics_returns['rmse']:.6f}

Next Steps:
  1. Use ml_ready_features.csv for advanced ML models
  2. Consider feature engineering (lags, interactions)
  3. Try non-linear models (Random Forest, XGBoost, Neural Networks)
  4. Implement walk-forward validation for time series
"""
print(summary_report)

with open(os.path.join(ml_data_dir, 'validation_summary.txt'), 'w') as f:
    f.write(summary_report)
print("OK: Saved validation summary: validation_summary.txt")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
