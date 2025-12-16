from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces

import numpy as np

@dataclass
class EnvConfig:
    transaction_cost: float = 1e-3
    reward_scale: float = 1.0
    episode_len: int = 252
    allow_short : bool = False
    include_prev_ret : bool = False
    include_grid_samples : bool = False
    grid_sample_k : int = 64
    eps_turnover : float = 0.2
    eps_drawdown : float = 0.2


class MarketPotentialEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, npz_path, config = EnvConfig(), seed = None):
        super().__init__()
        self.cfg = config

        data = np.load(npz_path, allow_pickle=False)
        self.X = data["X"].astype(np.float32)          
        self.phi = data["phi"].astype(np.float32)     
        self.ret = data["ret"].astype(np.float32)    
        self.tickers = data["tickers"].tolist() if "tickers" in data else None

        self.U_samp = None
        if self.cfg.include_grid_samples:
            self.U_samp = data["U_samp"].astype(np.float32)  

        self.T, self.N, _ = self.X.shape

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.N,), dtype=np.float32)

        obs_dim = self.N 
        obs_dim += 9 * self.N
        obs_dim += self.N 
        if self.cfg.include_prev_ret:
            obs_dim += self.N
        if self.cfg.include_grid_samples:
            obs_dim += self.cfg.grid_sample_k

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)

        self._t0: int = 0
        self._t: int = 0
        self._w: np.ndarray = np.ones(self.N, dtype=np.float32) / self.N
        self._w_prev: np.ndarray = self._w.copy()
        self._equity: float = 1.0

    def _action_to_weights(self, a):
        a = np.asarray(a, dtype=np.float32)

        if self.cfg.allow_short:
            w = np.tanh(a)
            denom = np.sum(np.abs(w)) + 1e-12
            return (w / denom).astype(np.float32)

        z = a - np.max(a)
        e = np.exp(z)
        w = e / (np.sum(e) + 1e-12)
        return w.astype(np.float32)

    def _get_obs(self):
        t = self._t
        parts = [
            self.phi[t].ravel(),
            self.X[t].reshape(-1),
            self._w.ravel(),
        ]
        if self.cfg.include_prev_ret:
            prev = self.ret[t - 1] if t > 0 else np.zeros(self.N, dtype=np.float32)
            parts.append(prev.ravel())
        if self.cfg.include_grid_samples:
            parts.append(self.U_samp[t][: self.cfg.grid_sample_k].ravel())
        return np.concatenate(parts, axis=0).astype(np.float32)

    def reset(self, *, seed = None, options = None):
        super().reset(seed = seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        ep_len = self.cfg.episode_len
        max_start = self.T - ep_len - 2

        if max_start < 1:
            raise RuntimeError("Not enough data for the configured episode_len.")

        self._t0 = int(self._rng.integers(1, max_start))
        self._t = self._t0

        self._w = np.ones(self.N, dtype=np.float32) / self.N
        self._w_prev = self._w.copy()
        self._equity = 1.0

        obs = self._get_obs()
        info = {"t0": self._t0, "tickers": self.tickers}
        return obs, info

    def step(self, action):
        self._w_prev = self._w
        self._w = self._action_to_weights(action)

        turnover = float(np.sum(np.abs(self._w - self._w_prev)))
        cost = self.cfg.transaction_cost * turnover

        r_next = self.ret[self._t + 1] 
        pnl = float(np.dot(self._w, r_next))

        reward = self.cfg.reward_scale * (pnl - cost) - self.cfg.eps_turnover * turnover - self.cfg.eps_drawdown * max(0.0, 1.0 - self._equity)

        self._equity *= float(1.0 + pnl - cost)
        self._t += 1

        terminated = False
        truncated = (self._t >= self._t0 + self.cfg.episode_len) or (self._t >= self.T - 2)

        obs = self._get_obs()
        info = {
            "t": self._t,
            "pnl": pnl,
            "cost": cost,
            "turnover": turnover,
            "equity": self._equity,
        }
        return obs, reward, terminated, truncated, info