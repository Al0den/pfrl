import numpy as np

def make_grid_2d(X, n = 60, margin = 1.0):
    assert X.shape[1] == 2
    x_min, x_max = X.iloc[:, 0].min(), X.iloc[:, 0].max()
    y_min, y_max = X.iloc[:, 1].min(), X.iloc[:, 1].max()

    x1 = np.linspace(x_min - margin, x_max + margin, n)
    x2 = np.linspace(y_min - margin, y_max + margin, n)

    xx, yy = np.meshgrid(x1, x2)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

    return grid, xx, yy

def make_grid_3d(X, n = 25, margin = 1.0):
    assert X.shape[1] == 3
    x_min, x_max = X.iloc[:, 0].min(), X.iloc[:, 0].max()
    y_min, y_max = X.iloc[:, 1].min(), X.iloc[:, 1].max()
    z_min, z_max = X.iloc[:, 2].min(), X.iloc[:, 2].max()

    x1 = np.linspace(x_min - margin, x_max + margin, n)
    x2 = np.linspace(y_min - margin, y_max + margin, n)
    x3 = np.linspace(z_min - margin, z_max + margin, n)

    xx, yy, zz = np.meshgrid(x1, x2, x3)
    grid = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    return grid, xx, yy, zz

