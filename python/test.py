import yfinance as yf
import pandas

from python.market_cache import MarketCache

tickers = ["AAPL"]
start = "2008-11-01"
end = "2025-12-01"

cache = MarketCache()
data = cache.get_prices(tickers, start, end, "1d")