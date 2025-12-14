import yfinance as yf
import pandas

from python.config import market, tickers
from python.math import matrice_correlation, weighted_graph_from_corr
from python.geometry import embedding_laplacian_eigenmaps

window_dates = market.get_trading_window_bounds(30, end="2025-12-12")

corr = matrice_correlation(tickers, window_dates)
W, D, L, G = weighted_graph_from_corr(tickers, corr)

X = embedding_laplacian_eigenmaps(W, d=2, normalized=True)
print(X.info())