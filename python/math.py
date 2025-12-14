from python.config import market, tickers, matrice_poids_alpha, matrice_poids_seuil

import numpy as np
import pandas as pd
import networkx as nx

def matrice_correlation(tickers, window_dates):
    dates = np.array(window_dates, dtype='datetime64[D]')
    start = dates.min()  
    end   = dates.max() + np.timedelta64(1, 'D')        

    data = market.get_prices(tickers, start, end, "1d")

    high = data.xs("High", level=1, axis=1) 

    log_high = np.log(high)
    returns = log_high.diff().dropna()

    corr = returns.corr()

    return corr

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
    W = matrice_poids(tickers, corr)

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


