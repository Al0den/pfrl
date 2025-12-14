# viz_graph.py

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from python.config import tickers, market
from python.math import matrice_correlation, weighted_graph_from_corr


def plot_correlation_graph(
    trading_days,
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
    seed: int = 42,
):
    """
    Construit le graphe de corrélation des tickers de `tickers` (déjà importés)
    sur la fenêtre `trading_days`, et le visualise avec un spring layout.

    ATTENDU dans le namespace :
        - tickers : liste de tickers
        - matrice_correlation(tickers, window_dates) -> DataFrame corr
        - weighted_graph_from_corr(tickers, corr) -> (W, D, L, G)
    """
    # Matrice de corrélation à partir de ta fonction existante
    corr = matrice_correlation(tickers, trading_days)

    # Construction du graphe pondéré (ta fonction existante)
    W, D, L, G = weighted_graph_from_corr(tickers, corr)

    if len(G.nodes) == 0:
        raise ValueError("Le graphe est vide (aucun nœud). Vérifie les tickers / données.")

    # Spring layout (les poids influencent la distance)
    pos = nx.spring_layout(G, weight="weight", seed=seed)

    # Récupération des poids d’arêtes pour les largeurs
    edges = list(G.edges(data=True))
    if edges:
        raw_weights = np.array([d.get("weight", 0.0) for (_, _, d) in edges])
        max_w = raw_weights.max() if raw_weights.size > 0 else 1.0

        # Normalisation pour des largeurs lisibles (entre 0.5 et 4 par ex.)
        widths = 0.5 + 3.5 * (raw_weights / max_w)
    else:
        raw_weights = np.array([])
        widths = []

    plt.figure(figsize=(10, 8))

    # Dessin des arêtes
    nx.draw_networkx_edges(
        G,
        pos,
        width=widths,
        alpha=0.6,
    )

    # Dessin des nœuds
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=400,
    )

    # Labels = tickers
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

    # On renvoie aussi les objets numériques si tu veux les réutiliser
    return {
        "corr": corr,
        "W": W,
        "D": D,
        "L": L,
        "G": G,
        "positions": pos,
    }


if __name__ == "__main__":
    # Exemple d’usage : tu peux adapter / supprimer ce bloc.
    # On suppose que tu as déjà fait les imports ailleurs, par ex :
    # from python.config import market, tickers
    # from ton_module_corr import matrice_correlation, weighted_graph_from_corr

    # Exemple : 30 derniers jours de bourse jusqu’à une date donnée
    trading_days = market.get_trading_window_bounds(5000, end="2025-12-12")

    print(trading_days)

    plot_correlation_graph(
        trading_days,
        title="Graphe de corrélation (30 derniers jours)",
        save_path="correlation_graph.png",
        show=True,
    )
