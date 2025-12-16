import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from collections import deque

from rl.env.market_potential_env import MarketPotentialEnv, EnvConfig

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, obs):
        mu = self.net(obs)
        std = torch.exp(self.log_std).clamp(1e-3, 2.0)
        return mu, std

def rollout_episode(env, policy, device, gamma = 0.99):
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    logps = []
    rewards = []
    infos = []

    done = False
    while not done:
        mu, std = policy(obs.unsqueeze(0))   
        mu = mu.squeeze(0)
        dist = torch.distributions.Normal(mu, std)

        a = dist.sample()
        logp = dist.log_prob(a).sum() 

        a_np = a.clamp(-1.0, 1.0).detach().cpu().numpy()

        obs_next, r, terminated, truncated, info = env.step(a_np)
        done = terminated or truncated

        logps.append(logp)
        rewards.append(float(r))
        infos.append(info)

        obs = torch.tensor(obs_next, dtype=torch.float32, device=device)

    R = 0.0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.append(R)
    returns.reverse()
    returns = torch.tensor(returns, dtype=torch.float32, device=device)

    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return logps, returns, rewards, infos

def max_drawdown(equity_curve):
    if len(equity_curve) == 0:
        return 0.0
    eq = np.asarray(equity_curve, dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-12)
    return float(dd.min())

@torch.no_grad()
def policy_action_stats(policy, obs_dim, act_dim, device):
    obs = torch.randn(256, obs_dim, device=device)
    mu, std = policy(obs)
    a = torch.distributions.Normal(mu, std).sample()
    a = a.clamp(-1.0, 1.0)
    a_np = a.detach().cpu().numpy()
    sat = np.mean(np.abs(a_np) > 0.98)
    return {
        "mu_mean": float(mu.mean().item()),
        "mu_std": float(mu.std().item()),
        "std_mean": float(std.mean().item()),
        "std_std": float(std.std().item()),
        "a_mean": float(a.mean().item()),
        "a_std": float(a.std().item()),
        "a_min": float(a.min().item()),
        "a_max": float(a.max().item()),
        "sat%": float(100.0 * sat),
    }

def grad_global_norm(parameters):
    sq = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach()
        sq += float(g.pow(2).sum().item())
    return float(np.sqrt(sq))

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    reward_ma = deque(maxlen=200)
    equity_ma = deque(maxlen=200)
    dd_ma = deque(maxlen=200)
    t0 = time.time()

    npz_path = "rl/data/features_train.npz"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MarketPotentialEnv(
        npz_path,
        EnvConfig(
            episode_len=252,
            transaction_cost=2e-3,
            include_prev_ret=True,
            include_grid_samples=False,
            grid_sample_k=64,
        ),
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = PolicyNet(obs_dim, act_dim, hidden=256).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    n_episodes = 50000
    gamma = 0.99

    print("Dataset length T =", env.T)

    for ep in range(1, n_episodes + 1):
        logps, returns, rewards, infos = rollout_episode(env, policy, device, gamma=gamma)

        logps_t = torch.stack(logps)
        loss = -(logps_t * returns).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        ep_reward = float(np.sum(rewards))
        ep_equity = infos[-1]["equity"] if infos else 1.0
        avg_turnover = float(np.mean([inf["turnover"] for inf in infos])) if infos else 0.0

        if ep % 10 == 0:
            ep_reward = float(np.sum(rewards))
            reward_ma.append(ep_reward)

            ep_equity = float(infos[-1].get("equity", 1.0)) if infos else 1.0
            equity_ma.append(ep_equity)

            avg_turnover = float(np.mean([inf.get("turnover", 0.0) for inf in infos])) if infos else 0.0
            max_turnover = float(np.max([inf.get("turnover", 0.0) for inf in infos])) if infos else 0.0

            eq_curve = [inf.get("equity", np.nan) for inf in infos if "equity" in inf]
            ep_dd = -max_drawdown(eq_curve) 
            dd_ma.append(ep_dd)

            probe = policy_action_stats(policy, obs_dim, act_dim, device)

            gnorm = grad_global_norm(policy.parameters())
            elapsed = time.time() - t0
            eps_per_sec = ep / max(elapsed, 1e-9)

            if ep % 10 == 0:
                print(
                    f"ep={ep:06d}  "
                    f"R={ep_reward:+.4f}  R_ma200={np.mean(reward_ma):+.4f}  "
                    f"eq={ep_equity:.4f}  eq_ma200={np.mean(equity_ma):.4f}  "
                    f"DD={ep_dd:.3%}  DD_ma200={np.mean(dd_ma):.3%}  "
                    f"to_avg={avg_turnover:.4f}  to_max={max_turnover:.4f}  "
                    f"loss={loss.item():+.6f}  gnorm={gnorm:.4f}  "
                    f"std_mean={probe['std_mean']:.3f}  sat={probe['sat%']:.1f}%  "
                    f"eps/s={eps_per_sec:.2f}"
                )

            if ep % 500 == 0:
                torch.save(policy.state_dict(), f"rl/data/reinforce_policy_ep{ep}.pt")
                print(f"[ckpt] saved rl/data/reinforce_policy_ep{ep}.pt")


    torch.save(policy.state_dict(), "rl/data/reinforce_policy.pt")
    print("[done] saved rl/data/reinforce_policy.pt")


if __name__ == "__main__":
    main()