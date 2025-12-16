import pandas as pd

def phi_momentum_score(returns):
    mom = returns.sum(axis = 0)
    phi = (mom - mom.mean()) / (mom.std() + 1e-12)
    return phi

def phi_risk_adjusted_momentum(returns, eps=1e-12):
    mu = returns.mean(axis=0)
    vol = returns.std(axis=0, ddof=1) + eps
    score = mu / vol 
    phi = (score - score.mean()) / (score.std() + eps)
    return phi


def phi_market_neutral_momentum(returns, market_ticker="^GSPC", eps=1e-12):
    m = returns[market_ticker].to_numpy()
    m = m - m.mean()

    phi_raw = {}
    for col in returns.columns:
        if col == market_ticker:
            continue
        y = returns[col].to_numpy()
        y = y - y.mean()

        beta = (y @ m) / (m @ m + eps)
        resid = y - beta * m

        phi_raw[col] = resid.sum()

    s = pd.Series(phi_raw)
    phi = (s - s.mean()) / (s.std() + eps)
    return phi.reindex(returns.columns).fillna(0.0)