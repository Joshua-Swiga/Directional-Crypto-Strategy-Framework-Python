import pandas as pd
import numpy as np
from env import smoothed_data
import matplotlib.pyplot as plt

# Load the smoothed data
df = pd.read_csv(smoothed_data)

# Ensure numeric types
df['open'] = df['open'].astype(float)
df['high'] = df['high'].astype(float)
df['low'] = df['low'].astype(float)
df['close'] = df['close'].astype(float)
df['volume'] = df['volume'].astype(float)
df['candle_body'] = df['candle_body'].astype(float)
df['candle_range'] = df['candle_range'].astype(float)
df['volatility'] = df['volatility'].astype(float)

# Select features for correlation analysis
features = ['open', 'high', 'low', 'close', 'volume', 'candle_body', 'candle_range', 'volatility']
df_features = df[features].copy()

print("=" * 80)
print("CORRELATION MATRIX - CURRENT FEATURES")
print("=" * 80)

# Compute correlation matrix
corr_matrix = df_features.corr()
print(corr_matrix)
print("\n")

# Create future target variables (shifted closes and returns)
df['future_close'] = df['close'].shift(-1)  # Next candle's close
df['returns'] = df['close'].pct_change()  # Current candle's returns
df['future_returns'] = df['future_close'].pct_change()  # Next candle's returns

# Compute correlations with future close prices
print("=" * 80)
print("CORRELATION WITH FUTURE CLOSE PRICE")
print("=" * 80)
corr_future_close = df[features].corrwith(df['future_close']).sort_values(ascending=False)
print(corr_future_close)
print("\n")

# Compute correlations with future returns
print("=" * 80)
print("CORRELATION WITH FUTURE RETURNS")
print("=" * 80)
corr_future_returns = df[features].corrwith(df['future_returns']).sort_values(ascending=False)
print(corr_future_returns)
print("\n")

# Interpretation and Analysis
print("=" * 80)
print("INTERPRETATION & ANALYSIS")
print("=" * 80)

# Find high correlations within features (redundancy)
print("\n1. REDUNDANT FEATURES (High Positive Correlation > 0.85):")
print("-" * 80)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.85:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

if high_corr_pairs:
    for feat1, feat2, corr_val in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True):
        print(f"  • {feat1} <-> {feat2}: {corr_val:.4f}")
        print(f"    → Consider dropping one of these features")
else:
    print("  • No highly correlated pairs found (> 0.85)")

# Find negative correlations (potential hedging or mean reversion)
print("\n2. NEGATIVE CORRELATIONS (Potential Mean Reversion or Hedging Signals):")
print("-" * 80)
neg_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] < -0.3:
            neg_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

if neg_corr_pairs:
    for feat1, feat2, corr_val in sorted(neg_corr_pairs, key=lambda x: x[2]):
        print(f"  • {feat1} <-> {feat2}: {corr_val:.4f}")
        print(f"    → Inverse relationship: potential mean reversion/hedging signal")
else:
    print("  • No strong negative correlations found (< -0.3)")

# Identify candidate predictors for regression
print("\n3. CANDIDATE PREDICTORS FOR REGRESSION (Significant correlations with future close/returns):")
print("-" * 80)
threshold = 0.15

good_predictors = corr_future_close[abs(corr_future_close) > threshold].sort_values(ascending=False)
print(f"\nPredictors for Future Close Price (|correlation| > {threshold}):")
for feat, corr_val in good_predictors.items():
    direction = "↑" if corr_val > 0 else "↓"
    print(f"  • {feat}: {corr_val:.4f} {direction}")

good_return_predictors = corr_future_returns[abs(corr_future_returns) > threshold].sort_values(ascending=False)
print(f"\nPredictors for Future Returns (|correlation| > {threshold}):")
for feat, corr_val in good_return_predictors.items():
    direction = "↑" if corr_val > 0 else "↓"
    print(f"  • {feat}: {corr_val:.4f} {direction}")

print("\n4. FEATURE STRENGTH SUMMARY:")
print("-" * 80)
avg_abs_corr = df[features].corrwith(df['future_close']).abs().sort_values(ascending=False)
for feat, strength in avg_abs_corr.items():
    if strength > 0.15:
        strength_level = "STRONG"
    elif strength > 0.10:
        strength_level = "MODERATE"
    else:
        strength_level = "WEAK"
    print(f"  • {feat}: {strength:.4f} ({strength_level})")

# Save correlation matrices for visualization
print("\n\n5. SAVING VISUALIZATION...")
print("-" * 80)

# Create correlation heatmap using matplotlib
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap 1: Feature correlations
im1 = axes[0].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
axes[0].set_xticks(range(len(corr_matrix.columns)))
axes[0].set_yticks(range(len(corr_matrix.columns)))
axes[0].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
axes[0].set_yticklabels(corr_matrix.columns)
axes[0].set_title('Feature Correlation Matrix')
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = axes[0].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha="center", va="center", color="black", fontsize=8)

# Heatmap 2: Correlations with future close
future_corr_df = pd.DataFrame({
    'Future Close': corr_future_close,
    'Future Returns': corr_future_returns
})
im2 = axes[1].imshow(future_corr_df, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
axes[1].set_xticks(range(len(future_corr_df.columns)))
axes[1].set_yticks(range(len(future_corr_df.index)))
axes[1].set_xticklabels(future_corr_df.columns)
axes[1].set_yticklabels(future_corr_df.index)
axes[1].set_title('Correlations with Future Price & Returns')
for i in range(len(future_corr_df.index)):
    for j in range(len(future_corr_df.columns)):
        text = axes[1].text(j, i, f'{future_corr_df.iloc[i, j]:.2f}', ha="center", va="center", color="black", fontsize=8)

plt.tight_layout()
plt.savefig(r'C:\Users\JoshuaSwiga\Desktop\strat\correlation_images\correlation_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved correlation plot to: correlation_analysis.png")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
