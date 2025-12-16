import numpy as np

from rl.env.market_potential_env import MarketPotentialEnv, EnvConfig

NPZ = "rl/data/features_train.npz"

def main():
    env = MarketPotentialEnv(
        NPZ,
        EnvConfig(
            episode_len=50,
            transaction_cost=1e-3,
            include_prev_ret=True,
            include_grid_samples=("U_samp" in np.load(NPZ).files),
            grid_sample_k=64,
        ),
        seed=123,
    )

    obs, info = env.reset()
    print("obs shape:", obs.shape)
    print("action shape:", env.action_space.shape)
    print("N:", env.N, "T:", env.T)

    total_r = 0.0
    for step in range(50):
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        total_r += r
        if (step + 1) % 10 == 0:
            print(f"step={step+1:03d} r={r:+.6f} equity={info['equity']:.4f} turnover={info['turnover']:.4f}")
        if terminated or truncated:
            print("done:", {"terminated": terminated, "truncated": truncated})
            break

    print("total reward:", total_r)

if __name__ == "__main__":
    main()
