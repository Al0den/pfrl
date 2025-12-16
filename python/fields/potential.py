import numpy as np
import pandas as pd

def potential_and_forces(X, phi, grid, sigma = 0.1):
    if isinstance(X, pd.DataFrame):
        X_values = X.to_numpy(dtype=float, copy=False)
        if isinstance(phi, (pd.Series, pd.DataFrame)):
            phi = phi.reindex(X.index)
    else:
        X_values = np.asarray(X, dtype=float)

    phi_values = np.asarray(phi, dtype=float).reshape(-1)
    grid_values = np.asarray(grid, dtype=float)

    if X_values.ndim != 2:
        raise ValueError(f"X must be 2D (N,d); got shape {X_values.shape}")
    if grid_values.ndim != 2:
        raise ValueError(f"grid must be 2D (M,d); got shape {grid_values.shape}")
    if grid_values.shape[1] != X_values.shape[1]:
        raise ValueError(
            f"grid and X must have same dimension d; got {grid_values.shape[1]} and {X_values.shape[1]}"
        )
    if phi_values.shape[0] != X_values.shape[0]:
        raise ValueError(
            f"phi must have length N={X_values.shape[0]}; got {phi_values.shape[0]}"
        )

    diff = grid_values[:, None, :] - X_values[None, :, :]
    dist2 = np.sum(diff ** 2, axis = 2)  # (M,N)

    K = np.exp(-dist2 / (2 * sigma ** 2))

    U = - K @ phi_values

    F = np.sum((phi_values[None, :, None] * (diff / (sigma**2)) * K[:, :, None]), axis=1)  # (M,d)

    return U, F
