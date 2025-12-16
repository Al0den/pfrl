from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

from time import sleep
import numpy as np
import pandas as pd
from tqdm import tqdm

from python.config import market, tickers
from python.math import matrice_correlation, weighted_graph_from_corr
from python.geometry import embedding_laplacian_eigenmaps, procrustes_align
from python.fields.signals import phi_risk_adjusted_momentum
from python.fields.potential import potential_and_forces

def _to_ts(x):
    return x if isinstance(x, pd.Timestamp) else pd.Timestamp(x)

def get_trading_days_from_cache(start, end, reference_ticker = "^GSPC"):
    df = market.get_prices(reference_ticker, start, end, interval="1d")

    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    idx = idx.normalize()

    days = sorted(set(idx.to_pydatetime()))

    return [pd.Timestamp(d) for d in days]

def compute_next_day_returns(days, tickers_list, start, end):

    df = market.get_prices(tickers_list, start, end, interval="1d")
    close = df.xs("Close", level=1, axis=1).copy()
   
    close.index = pd.to_datetime(close.index).normalize()
    close = close.groupby(close.index).last()
    close = close.sort_index()
    close = close.reindex(columns=tickers_list)

    ret = close.pct_change()

    next_days = [pd.Timestamp(d).normalize() for d in days[1:]]
    prev_days = [pd.Timestamp(d).normalize() for d in days[:-1]]

    rows = ret.reindex(next_days)
    
    keep = ~rows.isna().all(axis=1)
    if not keep.any():
        raise RuntimeError("Could not build next-day returns: all next-days missing/NaN after reindex.")

    rows = rows.loc[keep]
    out_days = [d for d, k in zip(prev_days, keep.to_numpy()) if k]

    ret_next = rows.to_numpy(dtype=np.float32)

    return ret_next, out_days


@dataclass
class EmbedFrame:
    date: pd.Timestamp
    X: np.ndarray     
    phi: np.ndarray    



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--reference", type=str, default="^GSPC")
    p.add_argument("--lookback", type=int, default=252)
    p.add_argument("--out", type=str, default="rl/data/features_train.npz")

    # U sampling.
    p.add_argument("--include_u_samp", action="store_true", help="Store U_samp (T,K) sampled from global grid.")
    p.add_argument("--u_k", type=int, default=64, help="Number of sampled U values per day if include_u_samp.")
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--grid", type=int, default=200, help="Global grid resolution (n x n) used ONLY for U sampling.")
    p.add_argument("--u_seed", type=int, default=42, help="Seed for selecting fixed probe points on the grid.")
    args = p.parse_args()

    
    start_ts = _to_ts(args.start).normalize()
    start_download_ts, _ = market.get_trading_window_bounds(args.lookback, end=start_ts)
    start_download_ts = start_download_ts.normalize() - pd.Timedelta(days=1)
    end_ts = _to_ts(args.end).normalize() + pd.Timedelta(days=1)

    tickers_list = list(tickers)
    N = len(tickers_list)

    market.get_prices(tickers_list, start_download_ts, end_ts, interval="1d")

    days = get_trading_days_from_cache(start_ts, end_ts, reference_ticker=args.reference)
    ret_next, ret_days = compute_next_day_returns(days, tickers_list, start=start_ts, end=end_ts)

    feat_days = ret_days
    T = len(feat_days)

    embed_frames = []
    X_prev = None

    xmin = ymin = +np.inf
    xmax = ymax = -np.inf

    for i, d in enumerate(tqdm(feat_days, desc="Building embeddings", unit="day")):
        window_start, window_end = market.get_trading_window_bounds(args.lookback, end=d)
        corr, returns = matrice_correlation(tickers_list, (window_start, window_end), return_returns=True)
        W, _, _, _ = weighted_graph_from_corr(tickers_list, corr)

        X_df = embedding_laplacian_eigenmaps(W, d=9, normalized=True)
        dim_cols = sorted(
            (c for c in X_df.columns if c.startswith("Dim_")),
            key=lambda c: int(c.split("_")[1])
        )
        X = X_df[dim_cols].to_numpy(dtype=float)

        if X_prev is not None:
            X = procrustes_align(X, X_prev)
        X_prev = X.copy()

        phi = phi_risk_adjusted_momentum(returns)
        phi = np.asarray(phi, dtype=float).reshape(-1)

        xmin = min(xmin, float(X[:, 0].min()))
        xmax = max(xmax, float(X[:, 0].max()))
        ymin = min(ymin, float(X[:, 1].min()))
        ymax = max(ymax, float(X[:, 1].max()))

        embed_frames.append(EmbedFrame(date=d, X=X.astype(np.float32), phi=phi.astype(np.float32)))

    X_arr = np.stack([e.X for e in embed_frames], axis=0).astype(np.float32)    
    phi_arr = np.stack([e.phi for e in embed_frames], axis=0).astype(np.float32)  
    dates_arr = np.array([e.date.strftime("%Y-%m-%d") for e in embed_frames], dtype="U10")

    save_dict = {
        "X": X_arr,
        "phi": phi_arr,
        "ret": ret_next,
        "dates": dates_arr,
        "tickers": np.array(tickers_list, dtype="U32"),
    }

    print("Feature arrays built: " + ", ".join(f"{k}: {v.shape}" for k, v in save_dict.items()))

    if args.include_u_samp:
        x1 = np.linspace(xmin - args.margin, xmax + args.margin, args.grid)
        x2 = np.linspace(ymin - args.margin, ymax + args.margin, args.grid)
        xx, yy = np.meshgrid(x1, x2)
        grid_global = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

        M = grid_global.shape[0]
        rng = np.random.default_rng(args.u_seed)
        K = int(args.u_k)
        if K <= 0 or K > M:
            raise ValueError(f"u_k must be in [1, {M}]")

        probe_idx = rng.choice(M, size=K, replace=False)
        save_dict["U_probe_idx"] = probe_idx.astype(np.int32)
        save_dict["U_grid_shape"] = np.array([args.grid, args.grid], dtype=np.int32)
        save_dict["U_grid_extent"] = np.array([float(xx.min()), float(xx.max()), float(yy.min()), float(yy.max())], dtype=np.float32)

        U_samp = np.zeros((T, K), dtype=np.float32)
        for t in tqdm(range(T), desc="Computing U_samp", unit="day"):
            U, _ = potential_and_forces(X_arr[t], phi_arr[t], grid_global)
            U_samp[t] = U.astype(np.float32)[probe_idx]
        save_dict["U_samp"] = U_samp

        print(f"Computed U_samp: {U_samp.shape}")

    out_path = args.out
    print(f"[save] {out_path}")
    np.savez_compressed(out_path, **save_dict)
    print("[done]")


if __name__ == "__main__":
    main()
