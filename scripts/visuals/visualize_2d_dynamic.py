#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

# Your project imports
from python.config import market, tickers  # market is your MarketCache instance
from python.math import matrice_correlation, weighted_graph_from_corr
from python.geometry import embedding_laplacian_eigenmaps
from python.fields.signals import phi_risk_adjusted_momentum
from python.fields.potential import potential_and_forces


# -----------------------------
# Helpers
# -----------------------------

def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found. Install it to export mp4.\n"
            "macOS: brew install ffmpeg\n"
            "Ubuntu: sudo apt-get install ffmpeg\n"
        )

def _to_ts(x) -> pd.Timestamp:
    return x if isinstance(x, pd.Timestamp) else pd.Timestamp(x)

def get_trading_days_from_cache(start: str, end: str, reference_ticker: str = "^GSPC") -> List[pd.Timestamp]:
    """
    Trading days are inferred from the cached daily bars of reference_ticker.
    This is consistent with your MarketCache logic.
    """
    start_ts = _to_ts(start).normalize()
    end_ts = _to_ts(end).normalize()

    # yfinance-style: start inclusive, end exclusive in your cache getter
    # So ask for end + 1 day to include the day `end` if it exists.
    df = market.get_prices(reference_ticker, start_ts, end_ts + pd.Timedelta(days=1), interval="1d")
    if df.empty:
        return []

    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    idx = idx.normalize()

    # unique + sorted
    days = sorted(set(idx.to_pydatetime()))
    return [pd.Timestamp(d) for d in days]

def procrustes_align(X_new: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    """
    Align X_new to X_ref with orthogonal Procrustes (+ translation).
    Prevents random rotations/reflections of eigenmaps across days.
    Shapes: (N,2).

    NOTE: You said you already modified this to remove reflections if needed.
    Keep your version here.
    """
    mu_new = X_new.mean(axis=0, keepdims=True)
    mu_ref = X_ref.mean(axis=0, keepdims=True)
    A = X_new - mu_new
    B = X_ref - mu_ref

    M = A.T @ B
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt

    # If you want to force "proper rotation" (no reflection), uncomment:
    # if np.linalg.det(R) < 0:
    #     U[:, -1] *= -1
    #     R = U @ Vt

    return (A @ R) + mu_ref


# -----------------------------
# Data containers
# -----------------------------

@dataclass
class EmbedFrame:
    end_date: pd.Timestamp
    X: np.ndarray          # (N,2) aligned
    phi: np.ndarray        # (N,)

@dataclass
class Keyframe:
    end_date: pd.Timestamp
    X: np.ndarray                    # (N,2) aligned
    U_grid: np.ndarray               # (n,n) on GLOBAL grid
    extent: Tuple[float, float, float, float]  # xmin, xmax, ymin, ymax (GLOBAL)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Create an MP4 of evolving potential landscape + moving points (MarketCache).")
    parser.add_argument("--start", type=str, required=True, help="First end-date in animation (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, required=True, help="Last end-date in animation (YYYY-MM-DD).")
    parser.add_argument("--reference", type=str, default="^GSPC", help="Reference ticker used to infer trading days.")
    parser.add_argument("--lookback", type=int, default=252, help="Lookback window in trading days.")
    parser.add_argument("--grid", type=int, default=200, help="Grid resolution n (n x n).")
    parser.add_argument("--margin", type=float, default=0.5, help="Grid margin around points.")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS.")
    parser.add_argument("--days_per_sec", type=float, default=5.0, help="Trading days per second.")
    parser.add_argument("--dpi", type=int, default=160, help="Output DPI.")
    parser.add_argument("--out", type=str, default="visuals/market_movie.mp4", help="Output mp4 path.")
    parser.add_argument("--no_labels", action="store_true", help="Disable ticker text labels (faster / cleaner).")
    parser.add_argument("--max_days", type=int, default=0, help="Debug: limit number of trading days.")
    args = parser.parse_args()

    ensure_ffmpeg()

    # 1) Trading day list (based on your cache)
    days = get_trading_days_from_cache(args.start, args.end, reference_ticker=args.reference)
    if args.max_days and args.max_days > 0:
        days = days[: args.max_days]

    if len(days) < 2:
        raise RuntimeError(
            f"Not enough trading days between {args.start} and {args.end}. "
            f"Check your cache coverage for {args.reference}."
        )

    frames_per_day = max(1, int(round(args.fps / args.days_per_sec)))
    print(f"[info] trading days: {len(days)}")
    print(f"[info] fps={args.fps}, days_per_sec={args.days_per_sec} -> frames_per_day={frames_per_day}")

    # ------------------------------------------------------------
    # 2) FIRST PASS: compute aligned embeddings + phi, gather global bounds
    # ------------------------------------------------------------
    embed_frames: List[EmbedFrame] = []
    X_prev: Optional[np.ndarray] = None

    xmin = ymin = +np.inf
    xmax = ymax = -np.inf

    for i, d in enumerate(days):
        print(f"[compute] {i+1}/{len(days)} end={d.date()}")

        window_start, window_end = market.get_trading_window_bounds(args.lookback, end=d)

        corr, returns = matrice_correlation(tickers, (window_start, window_end), return_returns=True)
        W, D, L, G = weighted_graph_from_corr(tickers, corr)

        X_df = embedding_laplacian_eigenmaps(W, d=2)
        X = np.column_stack([X_df["Dim_1"].to_numpy(), X_df["Dim_2"].to_numpy()]).astype(float)

        if X_prev is not None:
            X = procrustes_align(X, X_prev)
        X_prev = X.copy()

        phi = phi_risk_adjusted_momentum(returns)
        phi = np.asarray(phi, dtype=float).reshape(-1)

        xmin = min(xmin, float(X[:, 0].min()))
        xmax = max(xmax, float(X[:, 0].max()))
        ymin = min(ymin, float(X[:, 1].min()))
        ymax = max(ymax, float(X[:, 1].max()))

        embed_frames.append(EmbedFrame(end_date=d, X=X, phi=phi))

    # ------------------------------------------------------------
    # 3) Build ONE global grid (fixed across all frames)
    # ------------------------------------------------------------
    x1 = np.linspace(xmin - args.margin, xmax + args.margin, args.grid)
    x2 = np.linspace(ymin - args.margin, ymax + args.margin, args.grid)
    xx, yy = np.meshgrid(x1, x2)
    grid_global = np.stack([xx.ravel(), yy.ravel()], axis=1)

    GLOBAL_EXTENT = (float(xx.min()), float(xx.max()), float(yy.min()), float(yy.max()))

    # ------------------------------------------------------------
    # 4) SECOND PASS: compute U on the SAME grid for every day
    # ------------------------------------------------------------
    keyframes: List[Keyframe] = []
    global_min = +np.inf
    global_max = -np.inf

    for ef in embed_frames:
        U, _ = potential_and_forces(ef.X, ef.phi, grid_global)
        U_grid = U.reshape(xx.shape)

        global_min = min(global_min, float(np.nanmin(U_grid)))
        global_max = max(global_max, float(np.nanmax(U_grid)))

        keyframes.append(
            Keyframe(
                end_date=ef.end_date,
                X=ef.X,
                U_grid=U_grid,
                extent=GLOBAL_EXTENT,
            )
        )

    # ------------------------------------------------------------
    # 5) Figure setup
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    ax.set_xlim(GLOBAL_EXTENT[0], GLOBAL_EXTENT[1])
    ax.set_ylim(GLOBAL_EXTENT[2], GLOBAL_EXTENT[3])

    k0 = keyframes[0]
    im = ax.imshow(
        k0.U_grid,
        origin="lower",
        extent=[GLOBAL_EXTENT[0], GLOBAL_EXTENT[1], GLOBAL_EXTENT[2], GLOBAL_EXTENT[3]],
        aspect="auto",
        vmin=global_min,
        vmax=global_max,
    )
    sc = ax.scatter(k0.X[:, 0], k0.X[:, 1], s=35, color="black", zorder=5)

    texts = []
    if not args.no_labels:
        labels = [str(t) for t in tickers]
        texts = [
            ax.text(float(x), float(y), lab, fontsize=8, ha="left", va="bottom", zorder=6)
            for (x, y), lab in zip(k0.X, labels)
        ]

    ax.set_title(f"Potential landscape (2D) — end={k0.end_date.date()}")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    fig.colorbar(im, ax=ax, label="Potential U")
    plt.tight_layout()

    # ------------------------------------------------------------
    # 6) Video writer
    # ------------------------------------------------------------
    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=6000,
        extra_args=["-pix_fmt", "yuv420p"],
    )

    total_frames = (len(keyframes) - 1) * frames_per_day + 1
    print(f"[render] total frames={total_frames} -> {args.out}")

    with writer.saving(fig, args.out, dpi=args.dpi):
        pbar = tqdm(total=total_frames, desc="Rendering frames", unit="frame")

        # First frame
        writer.grab_frame()
        pbar.update(1)

        for i in range(len(keyframes) - 1):
            A = keyframes[i]
            B = keyframes[i + 1]

            for f in range(1, frames_per_day + 1):
                a = f / frames_per_day

                X = (1 - a) * A.X + a * B.X
                U_interp = (1 - a) * A.U_grid + a * B.U_grid

                sc.set_offsets(X)
                im.set_data(U_interp)

                if texts:
                    for txt, (x, y) in zip(texts, X):
                        txt.set_position((float(x), float(y)))

                ax.set_title(
                    f"Potential landscape (2D) — end={B.end_date.date()}  (interp {f}/{frames_per_day})"
                )

                writer.grab_frame()
                pbar.update(1)

        pbar.close()

    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
