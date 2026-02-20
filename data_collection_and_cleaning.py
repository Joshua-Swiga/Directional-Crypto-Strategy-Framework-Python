import pandas as pd
import numpy as np
from env import file_for_putting_data, smoothed_data
import os


def deriving_additional_features():
    if not os.path.exists(file_for_putting_data):
        print(f"Error: File '{file_for_putting_data}' does not exist. Run get_data_from_binance() first.")
        return

    df = pd.read_csv(file_for_putting_data)
    print(df.head(10))

    # Convert price columns to float
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    
    # Moving Average
    df['close_sma_5'] = df['close'].rolling(window=5).mean()
    
    # Candle Body (Close - Open)
    df['candle_body'] = df['close'] - df['open']
    
    # Candle Range (High - Low)
    df['candle_range'] = df['high'] - df['low']
    
    # Volatility (Rolling standard deviation of close prices)
    df['volatility'] = df['close'].rolling(window=10).std()
    
    # Bollinger Bands
    bb_period = 5
    bb_std = 2
    df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
    bb_std_dev = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
    df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)

    log_hl = np.log(df['high'] / df['low'])
    log_co = np.log(df['close'] / df['open'])

    df['garman_klass'] = np.sqrt(
        0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    )
    
    rsi_period = 5
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR (Average True Range)
    atr_period = 5
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=atr_period).mean()

    df.to_csv(smoothed_data, index=False)
    
    print("Feature derivation complete. Columns added:")
    print("- close_sma_5 (5-period SMA of close)")
    print("- candle_body (close - open)")
    print("- candle_range (high - low)")
    print(f"Volitility vectors: German-Klass (Individual stick), RSI (Volitility accross candles: {rsi_period} window period)")
    print("- volatility (5-period std of close)")
    print(f"- bb_middle, bb_upper, bb_lower ({bb_period}-period Bollinger Bands)")
    print("- garman_klass (OHLC volatility estimator)")
    print(f"- rsi ({rsi_period}-period RSI)")
    print(f"- atr ({atr_period}-period ATR)")
    print(f"Data has been saved to: {smoothed_data}")
    print("Because of fast reaction, noise is higher.")


if __name__ == "__main__":
    deriving_additional_features()

