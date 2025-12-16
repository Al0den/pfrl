from python.config import market, tickers
import numpy as np
import pandas as pd
start = pd.Timestamp("2017-01-01")
end   = pd.Timestamp("2025-12-01")
print("requested days from", start, "to", end)
print("tickers:", len(tickers))
df = market.get_prices(tickers, start, end, interval="1d")
print("get_prices index:", df.index.min(), "->", df.index.max(), "n=", len(df.index))
print("columns type:", type(df.columns), "ncols=", len(df.columns))