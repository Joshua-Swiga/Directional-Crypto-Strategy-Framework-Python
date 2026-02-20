import pandas as pd
smoothed_data_file = r"crypto info\smoothed_data.csv"


df = pd.read_csv(smoothed_data_file)
close_price = df['close_sma_10']
actual_close = df['close']
# print(
#     "Smoothed", close_price.tail(10)
#     )
print(
    'Actual', actual_close.tail()
)
delta = df['close'].diff()
print(delta.tail())