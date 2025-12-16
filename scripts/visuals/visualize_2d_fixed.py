import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.integrate import solve_ivp


from python.config import market, tickers
from python.math import matrice_correlation, weighted_graph_from_corr
from python.geometry import embedding_laplacian_eigenmaps
from python.fields.signals import phi_momentum_score, phi_risk_adjusted_momentum, phi_market_neutral_momentum
from python.fields.potential import potential_and_forces
from python.utilities.grid import make_grid_2d


# -----------------------------
# 0) Data + embedding + potential
# -----------------------------
window_dates = market.get_trading_window_bounds(60, end="2019-12-12")

corr, returns = matrice_correlation(tickers, window_dates, return_returns=True)
W, D, L, G = weighted_graph_from_corr(tickers, corr)

X = embedding_laplacian_eigenmaps(W, d=2, normalized=True)

phi = phi_risk_adjusted_momentum(returns)
grid, xx, yy = make_grid_2d(X, n=250, margin=0.5)  # n plus grand => plus lisse
U, F = potential_and_forces(X, phi, grid)

U_grid = U.reshape(xx.shape)  # (n,n)
Fx = F[:, 0].reshape(xx.shape)
Fy = F[:, 1].reshape(xx.shape)

xv = X["Dim_1"].to_numpy()
yv = X["Dim_2"].to_numpy()
labels = [str(t) for t in X.index]


# -----------------------------
# Figure + axes
# -----------------------------
fig = plt.figure(figsize=(12, 8))

# Axes 2D
ax2d = fig.add_subplot(111)
im = ax2d.imshow(
    U_grid,
    origin="lower",
    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
    aspect="auto",
)
sc = ax2d.scatter(xv, yv, s=35, color="black")
texts = [
    ax2d.text(x, y, lab, fontsize=8, ha="left", va="bottom")
    for x, y, lab in zip(xv, yv, labels)
]
ax2d.set_title("Potential landscape (2D flattened)")
ax2d.set_xlabel("Dim 1")
ax2d.set_ylabel("Dim 2")
cbar = fig.colorbar(im, ax=ax2d, label="Potential U")

# Axes 3D (créé mais caché)
ax3d = fig.add_subplot(111, projection="3d")
surf = ax3d.plot_surface(
    xx, yy, U_grid,
    cmap="viridis",
    linewidth=0,
    antialiased=True,
    alpha=0.95,
)
pts = ax3d.scatter(xv, yv, np.interp(xv, xx[0], U_grid.mean(axis=0)), color="black", s=25)
ax3d.scatter(
    xv, yv, np.interp(xv, xx[0], U_grid.mean(axis=0)),
    s=120,                 # plus gros
    c="black",             # noir pur
    edgecolors="black",    # pas de contour clair
    linewidths=0.0,
    depthshade=False,      # sinon Matplotlib éclaircit/assombrit
)
ax3d.set_title("Potential landscape (3D)")
ax3d.set_xlabel("Dim 1")
ax3d.set_ylabel("Dim 2")
ax3d.set_zlabel("Potential U")
ax3d.view_init(elev=35, azim=-135)
ax3d.set_visible(False)

ax_button = plt.axes([0.82, 0.02, 0.15, 0.05])
button = Button(ax_button, "Toggle 2D / 3D")

state = {"mode": "2d"}

def toggle(event):
    if state["mode"] == "2d":
        ax2d.set_visible(False)
        ax3d.set_visible(True)
        state["mode"] = "3d"
    else:
        ax3d.set_visible(False)
        ax2d.set_visible(True)
        state["mode"] = "2d"
    fig.canvas.draw_idle()

button.on_clicked(toggle)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.show()