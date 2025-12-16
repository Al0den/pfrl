#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from python.config import market, tickers
from python.math import matrice_correlation, matrice_poids_knn


def build_knn_graph(tickers_list, corr, k: int = 2, alpha: float = 1.0) -> tuple[nx.Graph, np.ndarray]:
    """
    Build an undirected weighted graph from correlation using KNN weights.
    - corr: pd.DataFrame (N,N)
    - returns: NetworkX graph and raw weights array for plotting widths
    """
    W = matrice_poids_knn(corr, k=k, alpha=alpha)  # <-- KNN here (default k=3)
    G = nx.Graph()
    for t in tickers_list:
        G.add_node(t)

    # Add edges where weight > 0
    for i in range(len(tickers_list)):
        for j in range(i + 1, len(tickers_list)):
            w = float(W.iloc[i, j])
            if w > 0:
                G.add_edge(tickers_list[i], tickers_list[j], weight=w)

    weights = np.array([d.get("weight", 0.0) for (_, _, d) in G.edges(data=True)], dtype=float)
    return G, weights


def main():
    p = argparse.ArgumentParser(description="Visualize correlation KNN graph (default k=3).")
    p.add_argument("--end", type=str, default="2025-12-12", help="End date (YYYY-MM-DD).")
    p.add_argument("--lookback", type=int, default=252, help="Lookback window in trading days.")
    p.add_argument("--k", type=int, default=3, help="K in KNN graph (default 3).")
    p.add_argument("--alpha", type=float, default=1.0, help="Weight exponent (W ** alpha).")
    p.add_argument("--seed", type=int, default=42, help="Layout seed.")
    p.add_argument("--layout", type=str, default="spring", choices=["spring", "kamada_kawai"], help="Graph layout.")
    p.add_argument("--title", type=str, default="", help="Figure title override.")
    p.add_argument("--save", type=str, default="", help="Save path (e.g. outputs/knn_graph.png).")
    p.add_argument("--show", action="store_true", help="Show interactive window.")
    args = p.parse_args()

    tickers_list = list(tickers)

    # Build correlation on rolling window
    w0, w1 = market.get_trading_window_bounds(args.lookback, end=args.end)
    corr = matrice_correlation(tickers_list, (w0, w1), return_returns=False)

    G, weights = build_knn_graph(tickers_list, corr, k=args.k, alpha=args.alpha)

    if args.layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G, weight="weight")
    else:
        pos = nx.spring_layout(G, weight="weight", seed=args.seed)

    # Edge widths scaled by weight
    if len(weights) > 0:
        wmax = float(np.max(weights))
        widths = 0.5 + 4.0 * (weights / (wmax + 1e-12))
    else:
        widths = []

    plt.figure(figsize=(12, 9))

    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, node_size=420)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.axis("off")
    title = args.title.strip()
    if not title:
        title = f"Correlation KNN Graph (k={args.k}) — lookback={args.lookback} — end={args.end}"
    plt.title(title)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=300)
        print(f"[saved] {args.save}")

    if args.show or not args.save:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
