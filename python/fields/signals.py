import pandas as pd

def phi_momentum_score(returns):
    mom = returns.sum(axis = 0)
    phi = (mom - mom.mean()) / (mom.std() + 1e-12)
    return phi