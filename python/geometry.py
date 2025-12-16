from python.config import tickers

import numpy as np
import pandas as pd

def laplacian_sym_normalized(W, eps = 1e-12):
    Wv = W.values.astype(float)
    d = Wv.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d + eps)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_sym = np.eye(Wv.shape[0]) - D_inv_sqrt @ Wv @ D_inv_sqrt
    return L_sym

def embedding_laplacian_eigenmaps(W, d = 2, normalized = True, tol = 1e-9, return_evals = False):
    tickers = list(W.index)
    N = len(tickers)

    if normalized:
        L = laplacian_sym_normalized(W)
    else:
        D = matrice_deg(W).values
        L = D - W.values
    
    evals, evecs = np.linalg.eigh(L)

    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    zero_mask = np.abs(evals) <= tol
    n_zeros = int(zero_mask.sum())

    nonzero = np.where(~zero_mask)[0]
    if len(nonzero) < d:
        raise ValueError(
            f"Not enough non-zero eigenvalues for d={d}. "
            f"Found only {len(nonzero)} > tol. Graph likely disconnected or threshold too high."
        )

    selected = nonzero[:d]
    embedding = evecs[:, selected]

    X_df = pd.DataFrame(embedding, index=tickers, columns=[f"Dim_{i+1}" for i in range(d)])
    
    if return_evals:
        return X_df, evals, n_zeros

    return X_df

def fix_signs(X_prev, X_curr):
    X = X_curr.copy()
    for k in range(X.shape[1]):
        if np.dot(X_prev[:, k], X[:, k]) < 0:
            X[:, k] = -X[:, k]
    return X

def procrustes_align(X_new: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    mu_new = X_new.mean(axis=0, keepdims=True)
    mu_ref = X_ref.mean(axis=0, keepdims=True)
    A = X_new - mu_new
    B = X_ref - mu_ref

    M = A.T @ B
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return (A @ R) + mu_ref

