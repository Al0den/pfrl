import pandas as pd

def potential_and_forces(X, phi, grid, sigma = 0.7):
    diff = grid[:, None, :] - X[None, :, :]
    dist2 = np.sum(diff ** 2, axis = 2)  # (M,N)

    K = np.exp(-dist2 / (2 * sigma ** 2))

    U = - K @ phi

    F = np.sum((phi[None, :, None] * (diff / (sigma**2)) * K[:, :, None]), axis=1)  # (M,d)

    return U, F