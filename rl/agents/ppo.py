#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from rl.env.market_potential_env import MarketPotentialEnv, EnvConfig


def atanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

def squash_log_det_jacobian(a: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(1.0 - a.pow(2) + eps).sum(dim=-1)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256,
                 log_std_min=-5.0, log_std_max=2.0):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)
        self.v_head = nn.Linear(hidden, 1)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def policy_params(self, obs):
        h = self.trunk(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mu, std

    def sample_action(self, obs):
        mu, std = self.policy_params(obs)
        dist = torch.distributions.Normal(mu, std)

        u = dist.rsample()
        a = torch.tanh(u)

        logp_u = dist.log_prob(u).sum(-1)
        log_det = squash_log_det_jacobian(a)
        logp = logp_u - log_det
        return a, logp

    def log_prob(self, obs, a):
        a = torch.clamp(a, -1.0 + 1e-6, 1.0 - 1e-6)
        u = atanh(a)
        mu, std = self.policy_params(obs)
        dist = torch.distributions.Normal(mu, std)
        return dist.log_prob(u).sum(-1) - squash_log_det_jacobian(a)

    def value(self, obs):
        h = self.trunk(obs)
        return self.v_head(h).squeeze(-1)


@torch.no_grad()
def compute_gae(rew, val, done, gamma=0.99, lam=0.95):
    T = rew.shape[0]
    adv = torch.zeros(T, device=rew.device)
    gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - done[t]
        delta = rew[t] + gamma * val[t + 1] * nonterminal - val[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    ret = adv + val[:-1]
    return adv, ret

@dataclass
class Rollout:
    obs: torch.Tensor
    act: torch.Tensor
    logp: torch.Tensor
    rew: torch.Tensor
    done: torch.Tensor
    val: torch.Tensor


def collect_rollout(env, ac, device, n_steps):
    obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []

    obs, _ = env.reset()
    for _ in range(n_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            a_t, logp_t = ac.sample_action(obs_t)
            v_t = ac.value(obs_t)

        a = a_t.squeeze(0).cpu().numpy()
        logp = logp_t.squeeze(0)
        v = v_t.squeeze(0)

        obs2, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        obs_buf.append(obs_t.squeeze(0))
        act_buf.append(a_t.squeeze(0))
        logp_buf.append(logp)
        rew_buf.append(torch.tensor(float(r), device=device))
        done_buf.append(torch.tensor(float(done), device=device))
        val_buf.append(v)

        obs = obs2
        if done:
            obs, _ = env.reset()

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        last_v = ac.value(obs_t).squeeze(0)
    val_buf.append(last_v)

    return Rollout(
        obs=torch.stack(obs_buf),
        act=torch.stack(act_buf),
        logp=torch.stack(logp_buf),
        rew=torch.stack(rew_buf),
        done=torch.stack(done_buf),
        val=torch.stack(val_buf),
    )

def ppo_update(ac, optimizer, rollout, adv, ret,
               clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
               train_iters=10, minibatch_size=512):

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    T = rollout.obs.shape[0]
    idx = np.arange(T)

    for _ in range(train_iters):
        np.random.shuffle(idx)
        for start in range(0, T, minibatch_size):
            mb = idx[start:start + minibatch_size]

            obs = rollout.obs[mb]
            act = rollout.act[mb]
            old_logp = rollout.logp[mb]
            mb_adv = adv[mb]
            mb_ret = ret[mb]

            new_logp = ac.log_prob(obs, act)
            ratio = torch.exp(new_logp - old_logp)

            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            pi_loss = -torch.min(surr1, surr2).mean()

            v = ac.value(obs)
            v_loss = 0.5 * (v - mb_ret).pow(2).mean()

            mu, std = ac.policy_params(obs)
            ent = torch.distributions.Normal(mu, std).entropy().sum(-1).mean()

            loss = pi_loss + vf_coef * v_loss - ent_coef * ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
            optimizer.step()


def compute_trading_metrics(infos, steps_per_year=252):
    pnl = np.array([x["pnl"] for x in infos])
    cost = np.array([x["cost"] for x in infos])
    turnover = np.array([x["turnover"] for x in infos])
    equity = np.array([x["equity"] for x in infos])

    r_net = pnl - cost
    T = len(r_net)

    cagr = (equity[-1]) ** (steps_per_year / T) - 1.0
    mu = r_net.mean()
    sig = r_net.std(ddof=1) + 1e-12

    sharpe = mu / sig * np.sqrt(steps_per_year)

    peak = np.maximum.accumulate(equity)
    mdd = np.max(1.0 - equity / peak)

    calmar = cagr / (mdd + 1e-12)

    cost_ratio = cost.sum() / (np.abs(pnl).sum() + 1e-12)
    to_avg = turnover.mean()

    score = sharpe + 0.25 * calmar - 0.5 * to_avg - 1.0 * cost_ratio

    return dict(
        eqT=equity[-1],
        cagr=cagr,
        sharpe=sharpe,
        mdd=mdd,
        calmar=calmar,
        to_avg=to_avg,
        cost_ratio=cost_ratio,
        score=score,
    )


@torch.no_grad()
def eval_trading(env, ac, device, n_episodes=5):
    ac.eval()
    metrics = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        infos = []
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mu, _ = ac.policy_params(obs_t)
            a = torch.tanh(mu).squeeze(0).cpu().numpy()
            obs, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            infos.append(info)
        metrics.append(compute_trading_metrics(infos))
    ac.train()

    return {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = EnvConfig(
        episode_len=252,
        transaction_cost=2e-3,
        include_prev_ret=True,
        eps_turnover=0.2,
        eps_drawdown=0.2,
    )

    env = MarketPotentialEnv("rl/data/features_train.npz", cfg, seed=0)
    eval_env = MarketPotentialEnv("rl/data/features_train.npz", cfg, seed=123)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    ac = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(ac.parameters(), lr=3e-4)

    steps_per_update = 4096
    updates = 20000
    best_score = -1e18

    pbar = trange(1, updates + 1, desc="PPO(trading)", dynamic_ncols=True)
    for upd in pbar:
        rollout = collect_rollout(env, ac, device, steps_per_update)
        adv, ret = compute_gae(rollout.rew, rollout.val, rollout.done)

        ppo_update(ac, optimizer, rollout, adv, ret)

        if upd % 10 == 0:
            m = eval_trading(eval_env, ac, device)
            pbar.write(
                f"[eval] upd={upd:06d} "
                f"score={m['score']:+.3f} "
                f"Sharpe={m['sharpe']:+.2f} "
                f"CAGR={m['cagr']:+.2%} "
                f"MDD={m['mdd']:.2%} "
                f"TO={m['to_avg']:.3f} "
                f"costR={m['cost_ratio']:.3f} "
                f"eqT={m['eqT']:.4f}"
            )
            if m["score"] > best_score:
                best_score = m["score"]
                torch.save(ac.state_dict(), "rl/data/ppo_tanh_best.pt")

    torch.save(ac.state_dict(), "rl/data/ppo_tanh.pt")


if __name__ == "__main__":
    main()
