import requests
from datetime import datetime
import pandas as pd
import time

from env import (
    data_url,
    file_for_putting_data,
    SYMBOL,
    INTERVAL,        
    RAW_INTERVAL,    
    TOTAL_CANDLES,
    MAX_LIMIT_PER_REQUEST,
)


def get_data_from_binance(
    symbol: str = SYMBOL,
    raw_interval: str = RAW_INTERVAL,
    target_interval: str = INTERVAL,
    total_candles: int = TOTAL_CANDLES,
):
    """
    Download latest raw candles from Binance, resample to `target_interval`, and write to collected_data.csv.

    - Fetch RAW_INTERVAL (e.g. 5m) klines from Binance
    - Resample locally into 20m bars
    - Keep the most recent `total_candles` 20m bars
    """
    print(f"Attempting to get {total_candles} {target_interval} candles from Binance ({symbol}, raw={raw_interval})...")

    url_base = "https://api.binance.com/api/v3/klines"
    limit = min(MAX_LIMIT_PER_REQUEST, 1000)

    # For 20m from 5m we need roughly 4x raw bars
    raw_needed = total_candles * 4

    all_rows = []
    end_time = None  # ms

    while len(all_rows) < raw_needed:
        params = {"symbol": symbol, "interval": raw_interval, "limit": limit}
        if end_time is not None:
            params["endTime"] = end_time

        resp = requests.get(url_base, params=params, timeout=30)
        resp.raise_for_status()
        chunk = resp.json()
        if not chunk:
            break

        # oldest -> newest
        all_rows.extend(chunk)

        first_open_time = chunk[0][0]
        end_time = int(first_open_time) - 1

        if len(chunk) < limit:
            break

    if not all_rows:
        raise RuntimeError("No data returned from Binance for klines.")

    # Build DataFrame from raw klines
    cols = [
        "open_time_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time_ms",
        "quote_volume",
        "num_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df_raw = pd.DataFrame(all_rows, columns=cols)

    df_raw["open_time"] = pd.to_datetime(df_raw["open_time_ms"], unit="ms")
    df_raw["close_time"] = pd.to_datetime(df_raw["close_time_ms"], unit="ms")
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
    df_raw["num_trades"] = pd.to_numeric(df_raw["num_trades"], errors="coerce").astype("Int64")

    df_raw = df_raw.sort_values("open_time")

    # Resample to target 20m bars
    df_res = (
        df_raw.set_index("open_time")
        .resample("20min")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "quote_volume": "sum",
                "num_trades": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )

    df_res["close_time"] = df_res["open_time"] + pd.to_timedelta(20, unit="m") - pd.to_timedelta(1, unit="s")

    df_res = df_res.tail(total_candles).copy()

    cols_out = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "num_trades",
    ]
    open(file_for_putting_data, "w").close()
    print("Initial data has been cleared. Appending new records.")
    time.sleep(3)
    df_res.to_csv(file_for_putting_data, index=False, columns=cols_out)
    print(f"Data has been written to: {file_for_putting_data}. Note: Current trades are last in the file. We are using a {target_interval} timeframe.")
