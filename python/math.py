from python.config import market, tickers, matrice_poids_alpha, matrice_poids_seuil

import numpy as np
import pandas as pd
import networkx as nx

def matrice_correlation(tickers, window_dates, return_returns = False):
    dates = np.array(window_dates, dtype='datetime64[D]')
    start = dates.min()  
    end   = dates.max() + np.timedelta64(1, 'D')        

    data = market.get_prices(tickers, start, end, "1d")

    close = data.xs("Close", level=1, axis=1) 

    log_close = np.log(close)
    returns = log_close.diff().dropna(how="all")
    corr = returns.corr()

    if return_returns:
        return corr, returns

    return corr

def matrice_poids_knn(corr, k=8, alpha=1.0):
    C = corr.copy()
    np.fill_diagonal(C.values, 0.0)
    W = np.zeros_like(C.values)
    for i in range(C.shape[0]):
        idx = np.argsort(-C.values[i])[:k]  # top-k corr (signées ou positives)
        W[i, idx] = np.maximum(C.values[i, idx], 0.0)  # ou garde le signe si tu sais gérer
    W = (W + W.T) / 2
    W = W ** alpha
    return pd.DataFrame(W, index=C.index, columns=C.columns)

def matrice_poids(tickers, corr, alpha=matrice_poids_alpha, seuil=matrice_poids_seuil):
    W = abs(corr) ** alpha * (abs(corr) >= seuil)
    np.fill_diagonal(W.values, 0)

    W = W.loc[tickers, tickers]

    return W

def matrice_deg(W):
    D = pd.DataFrame(np.diag(W.sum(axis=1)), index=W.index, columns=W.columns)
    return D

def matrice_laplacien(W, D):
    return D - W

def weighted_graph_from_corr(tickers, corr):
    W = matrice_poids_knn(corr)

    D = matrice_deg(W)
    L = matrice_laplacien(W, D)

    G = nx.Graph()

    for t in tickers:
        G.add_node(t)

    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            w_ij = W.iloc[i, j]
            if w_ij > 0:
                G.add_edge(tickers[i], tickers[j], weight=w_ij)

    return W, D, L, G

if __name__ == "__main__":
    mat_corr = matrice_correlation(tickers, ("2018-01-01", "2025-12-01"))

