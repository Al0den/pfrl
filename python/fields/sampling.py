import numpy as np

def make_grid_2d(X, n = 60, margin = 1.0):
    assert X.shape[1] == 2
    x_min, x_max = X.iloc[:, 0].min(), X.iloc[:, 0].max()
    y_min, y_max = X.iloc[:, 1].min(), X.iloc[:, 1].max()

    x1 = np.linspace(x_min - margin, x_max + margin, n)
    x2 = np.linspace(y_min - margin, y_max + margin, n)

    xx, yy = np.meshgrid(x1, x2)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (M,2)

    return grid, xx, yy