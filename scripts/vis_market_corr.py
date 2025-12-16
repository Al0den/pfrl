# viz_graph.py

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from python.config import tickers, market
from python.math import matrice_correlation, weighted_graph_from_corr


def plot_correlation_graph(trading_days, title = None, save_pat = None,show = True, seed = 42,):  
    corr = matrice_correlation(tickers, trading_days)

    W, D, L, G = weighted_graph_from_corr(tickers, corr)

    pos = nx.spring_layout(G, weight="weight", seed=seed)

    edges = list(G.edges(data=True))
    if edges:
        raw_weights = np.array([d.get("weight", 0.0) for (_, _, d) in edges])
        max_w = raw_weights.max() if raw_weights.size > 0 else 1.0

        widths = 0.5 + 3.5 * (raw_weights / max_w)
    else:
        raw_weights = np.array([])
        widths = []

    plt.figure(figsize=(10, 8))

    nx.draw_networkx_edges(
        G,
        pos,
        width=widths,
        alpha=0.6,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=400,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=9,
    )

    plt.axis("off")
    if title is None:
        title = "Graphe de corrélation des tickers"
    plt.title(title)

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

    return {
        "corr": corr,
        "W": W,
        "D": D,
        "L": L,
        "G": G,
        "positions": pos,
    }


if __name__ == "__main__":
    trading_days = market.get_trading_window_bounds(5000, end="2025-12-12")

    print(trading_days)

    plot_correlation_graph(
        trading_days,
        title="Graphe de corrélation (30 derniers jours)",
        save_path="correlation_graph.png",
        show=True,
    )
