SYMBOL = "BTCUSDT"

# Target trading timeframe (your sticks)
INTERVAL = "20m"

# Raw Binance API interval we fetch and then resample
RAW_INTERVAL = "5m"  # valid Binance interval; we aggregate 4x into 20m

MAX_LIMIT_PER_REQUEST = 1000
TOTAL_CANDLES = 10000  # keep a rolling window of the latest 10,000 target candles

# Base URL (still built from RAW_INTERVAL; resampling happens locally)
data_url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval={RAW_INTERVAL}&limit={MAX_LIMIT_PER_REQUEST}"
file_for_putting_data = r'C:\Users\JoshuaSwiga\Desktop\strat\crypto info\collected_data.csv'
smoothed_data = r"C:\Users\JoshuaSwiga\Desktop\strat\crypto info\smoothed_data.csv" 