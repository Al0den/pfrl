import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from python.config import market, tickers
from python.math import matrice_correlation, weighted_graph_from_corr
from python.geometry import embedding_laplacian_eigenmaps
from python.fields.signals import phi_momentum_score
from python.fields.potential import potential_and_forces
from python.fields.sampling import make_grid_2d

window_dates = market.get_trading_window_bounds(30, end="2025-12-12")

corr, returns = matrice_correlation(tickers, window_dates, return_returns=True)
W, D, L, G = weighted_graph_from_corr(tickers, corr)

X = embedding_laplacian_eigenmaps(W, d=2, normalized=True)
print(X.info())

phi = phi_momentum_score(returns)
grid, xx, yy = make_grid_2d(X)
U, F = potential_and_forces(X, phi, grid)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, U.reshape(xx.shape), levels=50, cmap='viridis')
plt.colorbar(label='Potential U')
plt.quiver(xx, yy, F[:, 0].reshape(xx.shape), F[:, 1].reshape(yy.shape), color='white', alpha=0.6)
plt.scatter(X['Dim_1'], X['Dim_2'], c='red', edgecolor='k')
plt.title('2D Embedding with Potential Field and Forces')
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.show()


