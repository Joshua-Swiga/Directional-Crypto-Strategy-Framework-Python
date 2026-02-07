import requests
from datetime import datetime 
import pandas as pd
from env import data_url, file_for_putting_data, smoothed_data
import os
def get_data_from_binance():
    response = requests.get(data_url)
    data = response.json()[::-1]
    columns=[
        'open_time','open','high','low','close','volume',
        'close_time','quote_volume','num_trades',
        ]

    with open(file_for_putting_data, 'w') as f:
        f.write(','.join(columns) + '\n')


    with open(file_for_putting_data, 'a') as f:
        for item in data:
            open_time = datetime.fromtimestamp(item[0] / 1000)  
            close_time = datetime.fromtimestamp(item[6] / 1000)
            open_price = float(item[1])
            high_price = float(item[2])
            low_price = float(item[3])
            close_price = float(item[4])
            volume = float(item[5])
            quote_volume = float(item[7])
            num_trades = int(item[8])
            
            f.write(f"{open_time},{open_price},{high_price},{low_price},{close_price},{volume},{close_time},{quote_volume},{num_trades}\n")
        print("Data has been written to the file. You can check the file now.")
# get_data_from_binance()
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
    df['close_sma_10'] = df['close'].rolling(window=10).mean()
    
    # Candle Body (Close - Open)
    df['candle_body'] = df['close'] - df['open']
    
    # Candle Range (High - Low)
    df['candle_range'] = df['high'] - df['low']
    
    # Volatility (Rolling standard deviation of close prices)
    df['volatility'] = df['close'].rolling(window=10).std()

    df.to_csv(smoothed_data, index=False)
    print(f"Data has been smoothed and saved to {smoothed_data}")

deriving_additional_features()