from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from rl.env import PinningEnv, EnvConfig, make_env

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


class MetricsCallback(BaseCallback):
    def __init__(self, eval_env: PinningEnv, eval_freq: int = 4096, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.eval_results = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        metrics = evaluate_policy(self.eval_env, self.model, n_episodes=50)
        self.eval_results.append({"timesteps": self.num_timesteps, **metrics})

        if self.verbose > 0:
            print(f"  [Step {self.num_timesteps:>7d}] "
                  f"pnl={metrics['mean_pnl']:+.3f}% "
                  f"win={metrics['win_rate']:.1f}% "
                  f"sharpe={metrics['sharpe']:.3f} "
                  f"trades/ep={metrics['mean_trades']:.2f} "
                  f"flat_rate={metrics['flat_rate']:.1f}%")

        if metrics["mean_pnl"] > self.best_mean_reward:
            self.best_mean_reward = metrics["mean_pnl"]
            path = MODEL_DIR / "ppo_pinning_best"
            self.model.save(str(path))
            if self.verbose > 0:
                print(f"    -> New best model saved (pnl={metrics['mean_pnl']:+.3f}%)")

        return True


def evaluate_policy(env: PinningEnv, model, n_episodes: int = 50) -> dict:
    pnls = []
    n_trades_list = []
    flat_actions = 0
    total_actions = 0

    for i in range(n_episodes):
        obs, info = env.reset(seed=None)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if action == 1:
                flat_actions += 1
            total_actions += 1
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        pnls.append(info.get("episode_pnl", 0.0))
        n_trades_list.append(info.get("n_trades", 0))

    pnls = np.array(pnls)
    n_trades_arr = np.array(n_trades_list)
    flat_rate = flat_actions / total_actions * 100 if total_actions > 0 else 0

    sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(52) if np.std(pnls) > 0 else 0
    win_rate = np.mean(pnls > 0) * 100

    return {
        "mean_pnl": float(np.mean(pnls)),
        "std_pnl": float(np.std(pnls)),
        "median_pnl": float(np.median(pnls)),
        "win_rate": float(win_rate),
        "sharpe": float(sharpe),
        "mean_trades": float(np.mean(n_trades_arr)),
        "flat_rate": float(flat_rate),
        "n_episodes": n_episodes,
    }


def train_ppo(
    config: EnvConfig | None = None,
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    seed: int = 42,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    net_arch: list | None = None,
    eval_freq: int = 4096,
    verbose: int = 1,
    tag: str = "",
):
    if config is None:
        config = EnvConfig()
    if net_arch is None:
        net_arch = [128, 128, 64]

    if verbose > 0:
        print(f"Training PPO | tag={tag} | seed={seed} | n_envs={n_envs}")
        print(f"  total_timesteps={total_timesteps}")
        print(f"  net_arch={net_arch}, lr={learning_rate}")
        print(f"  config: etf_only={config.etf_only}, min_dte={config.min_dte}, max_dte={config.max_dte}")

    def make_train_env(rank):
        def _init():
            env = PinningEnv(config=config, split="train")
            env.reset(seed=seed + rank)
            return env
        return _init

    train_envs = DummyVecEnv([make_train_env(i) for i in range(n_envs)])

    eval_env = PinningEnv(config=config, split="val")

    policy_kwargs = dict(net_arch=net_arch)

    model = PPO(
        "MlpPolicy",
        train_envs,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
        device="cpu",
    )

    callback = MetricsCallback(eval_env=eval_env, eval_freq=eval_freq, verbose=verbose)

    if verbose > 0:
        print(f"\nStarting training...")

    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    tag_suffix = f"_{tag}" if tag else ""
    final_path = MODEL_DIR / f"ppo_pinning_final{tag_suffix}"
    model.save(str(final_path))
    if verbose > 0:
        print(f"\nFinal model saved to {final_path}")

    if callback.eval_results:
        eval_df = pd.DataFrame(callback.eval_results)
        eval_path = DATA_DIR / f"rl_eval{tag_suffix}.parquet"
        eval_df.to_parquet(eval_path, index=False)
        if verbose > 0:
            print(f"Eval results saved to {eval_path}")

    train_envs.close()

    return model, callback.eval_results


def run_ensemble(
    config: EnvConfig | None = None,
    n_seeds: int = 5,
    total_timesteps: int = 500_000,
    **kwargs,
) -> list:
    if config is None:
        config = EnvConfig()

    all_models = []
    all_results = []

    for seed in range(n_seeds):
        print(f"\n{'='*60}")
        print(f"  Seed {seed + 1}/{n_seeds}")
        print(f"{'='*60}")

        model, eval_results = train_ppo(
            config=config,
            total_timesteps=total_timesteps,
            seed=seed * 100,
            tag=f"seed{seed}",
            **kwargs,
        )
        all_models.append(model)
        all_results.append(eval_results)

    test_env = PinningEnv(config=config, split="test")

    print(f"\n{'='*60}")
    print(f"  Ensemble Test Results ({n_seeds} seeds)")
    print(f"{'='*60}")

    seed_metrics = []
    for i, model in enumerate(all_models):
        m = evaluate_policy(test_env, model, n_episodes=len(test_env.episodes))
        seed_metrics.append(m)
        print(f"  Seed {i}: pnl={m['mean_pnl']:+.3f}% win={m['win_rate']:.1f}% "
              f"sharpe={m['sharpe']:.3f} trades/ep={m['mean_trades']:.2f}")

    agg = {}
    for key in ["mean_pnl", "std_pnl", "win_rate", "sharpe", "mean_trades", "flat_rate"]:
        vals = [m[key] for m in seed_metrics]
        agg[key] = {"mean": np.mean(vals), "std": np.std(vals)}

    print(f"\n  Ensemble average:")
    for key, v in agg.items():
        print(f"    {key}: {v['mean']:.4f} ± {v['std']:.4f}")

    ensemble_df = pd.DataFrame(seed_metrics)
    ensemble_df.to_parquet(DATA_DIR / "rl_ensemble_test.parquet", index=False)

    return all_models, agg


def compare_with_baseline(config: EnvConfig | None = None, verbose: bool = True):
    if config is None:
        config = EnvConfig()

    test_env = PinningEnv(config=config, split="test")

    rule_pnls = []
    for ep_idx in range(len(test_env.episodes)):
        obs, info = test_env.reset()
        done = False
        while not done:
            row = test_env._ep_data.iloc[test_env._step]
            spot = float(row["spot_price"])
            target = float(row.get("max_volume_strike", np.nan))
            if pd.isna(target) or spot <= 0:
                action = 1
            else:
                dist_pct = (target - spot) / spot * 100
                if dist_pct > 0.5:
                    action = 2
                elif dist_pct < -0.5:
                    action = 0
                else:
                    action = 1
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
        rule_pnls.append(info.get("episode_pnl", 0.0))

    rule_pnls = np.array(rule_pnls)
    rule_sharpe = np.mean(rule_pnls) / np.std(rule_pnls) * np.sqrt(52) if np.std(rule_pnls) > 0 else 0

    if verbose:
        print(f"\nRule-based baseline on test set:")
        print(f"  Trades: {len(rule_pnls)}")
        print(f"  Mean PnL: {np.mean(rule_pnls):+.4f}%")
        print(f"  Win Rate: {np.mean(rule_pnls > 0) * 100:.1f}%")
        print(f"  Sharpe: {rule_sharpe:.3f}")

    return {
        "mean_pnl": float(np.mean(rule_pnls)),
        "std_pnl": float(np.std(rule_pnls)),
        "win_rate": float(np.mean(rule_pnls > 0) * 100),
        "sharpe": float(rule_sharpe),
    }


if __name__ == "__main__":
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    config = EnvConfig(
        etf_only=True,
        min_dte=1,
        max_dte=10,
        slippage_bps=5.0,
        commission_bps=1.0,
        reward_scaling=100.0,
        penalty_no_trade=0.5,
    )

    print("\n--- Rule-based Baseline ---")
    baseline = compare_with_baseline(config=config)

    print("\n--- Training Single PPO ---")
    model, eval_results = train_ppo(
        config=config,
        total_timesteps=200_000,
        n_envs=4,
        seed=42,
        eval_freq=4096,
        tag="single",
    )

    test_env = PinningEnv(config=config, split="test")
    rl_metrics = evaluate_policy(test_env, model, n_episodes=min(200, len(test_env.episodes)))
    print(f"\n--- RL vs Baseline Comparison ---")
    print(f"  Baseline: pnl={baseline['mean_pnl']:+.4f}% sharpe={baseline['sharpe']:.3f}")
    print(f"  RL:       pnl={rl_metrics['mean_pnl']:+.4f}% sharpe={rl_metrics['sharpe']:.3f}")

    print("\n--- Training Ensemble (5 seeds) ---")
    models, agg = run_ensemble(
        config=config,
        n_seeds=5,
        total_timesteps=200_000,
        n_envs=4,
    )
