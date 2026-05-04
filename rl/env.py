from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from dataclasses import dataclass


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ETF_TICKERS = {"SPY", "QQQ", "GLD"}

STATE_COLUMNS = [
    "oi_concentration", "call_put_ratio", "oi_skew", "oi_kurtosis",
    "oi_vol_ratio", "total_oi", "total_volume", "num_contracts",
    "days_to_expiry", "is_monthly",
    "price_vs_sma5_pct", "price_vs_sma20_pct", "price_vs_sma50_pct", "price_vs_sma200_pct",
    "rsi_14", "atr_pct", "return_5d", "realized_vol_20d",
    "dist_to_resistance_pct", "dist_to_support_pct",
    "atm_iv",
    "max_volume_strike", "max_oi_strike", "max_call_oi_strike",
    "max_put_oi_strike", "max_net_oi_strike", "oi_mass_center", "oi_top1_strike",
]

TICKER_MAP = {t: i for i, t in enumerate(sorted([
    "AAPL", "AMZN", "GLD", "GOOG", "GOOGL", "MSFT", "NVDA", "QQQ", "SPY"
]))}


@dataclass
class EnvConfig:
    tickers: list[str] | None = None
    etf_only: bool = True
    train_end: str = "2019-12-31"
    val_end: str = "2022-12-31"
    min_dte: int = 1
    max_dte: int = 10
    slippage_bps: float = 5.0
    commission_bps: float = 1.0
    reward_scaling: float = 100.0
    penalty_no_trade: float = 0.0


def _load_features() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    df = df.sort_values(["ticker", "expiration", "quote_date"]).reset_index(drop=True)
    return df


def _normalize_strike(strike: float | np.ndarray, spot: float) -> float | np.ndarray:
    return (strike - spot) / spot


class PinningEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None, split: str = "train"):
        super().__init__()
        self.config = config or EnvConfig()
        self.split = split

        self._state_cols = STATE_COLUMNS
        n_ticker = len(TICKER_MAP)
        self.observation_dim = len(self._state_cols) + n_ticker + 2
        self.n_tickers = n_ticker

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.observation_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self._prepare_episodes()

    def _prepare_episodes(self):
        df = _load_features()

        if self.config.tickers:
            df = df[df["ticker"].isin(self.config.tickers)]
        elif self.config.etf_only:
            df = df[df["ticker"].isin(ETF_TICKERS)]

        if self.split == "train":
            df = df[df["quote_date"] <= self.config.train_end]
        elif self.split == "val":
            df = df[(df["quote_date"] > self.config.train_end) & (df["quote_date"] <= self.config.val_end)]
        elif self.split == "test":
            df = df[df["quote_date"] > self.config.val_end]

        self.episodes = []
        for (ticker, expiration), group in df.groupby(["ticker", "expiration"]):
            group = group.sort_values("quote_date").reset_index(drop=True)
            valid = group[
                (group["days_to_expiry"] >= self.config.min_dte)
                & (group["days_to_expiry"] <= self.config.max_dte)
            ]
            if valid.empty:
                continue
            valid = valid.copy()
            valid["_ticker_idx"] = TICKER_MAP.get(ticker, 0)

            expiry_row = group[group["days_to_expiry"] == group["days_to_expiry"].min()]
            if expiry_row.empty:
                continue
            expiry_close = float(expiry_row.iloc[0]["expiry_close"])
            if np.isnan(expiry_close) or expiry_close <= 0:
                continue

            self.episodes.append({
                "ticker": ticker,
                "expiration": expiration,
                "data": valid,
                "n_steps": len(valid),
                "expiry_close": expiry_close,
            })

        self._episode_order = list(range(len(self.episodes)))
        self._rng = np.random.default_rng()
        self._shuffle()

    def _shuffle(self):
        self._rng.shuffle(self._episode_order)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._shuffle()

        if not self._episode_order:
            self._prepare_episodes()

        self._ep_idx = self._episode_order.pop()
        ep = self.episodes[self._ep_idx]
        self._ep_data = ep["data"]
        self._ep_steps = ep["n_steps"]
        self._expiry_close = ep["expiry_close"]
        self._step = 0
        self._position = 0
        self._entry_price = 0.0
        self._total_reward = 0.0
        self._total_pnl = 0.0
        self._n_trades = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_obs(self) -> np.ndarray:
        row = self._ep_data.iloc[self._step]
        spot = float(row["spot_price"])

        features = []
        for col in self._state_cols:
            val = row.get(col, np.nan)
            if col in ("is_monthly",):
                val = float(val)
            elif col in ("max_volume_strike", "max_oi_strike", "max_call_oi_strike",
                         "max_put_oi_strike", "max_net_oi_strike", "oi_mass_center",
                         "oi_top1_strike"):
                val = _normalize_strike(float(val), spot) if pd.notna(val) and spot > 0 else 0.0
            elif col == "days_to_expiry":
                val = float(val) / 10.0
            elif col in ("total_oi", "total_volume", "num_contracts"):
                val = np.log1p(float(val)) if pd.notna(val) else 0.0
            else:
                val = float(val) if pd.notna(val) else 0.0
            features.append(val)

        ticker_onehot = np.zeros(self.n_tickers, dtype=np.float32)
        ticker_idx = int(row.get("_ticker_idx", 0))
        if ticker_idx < self.n_tickers:
            ticker_onehot[ticker_idx] = 1.0
        features.extend(ticker_onehot.tolist())

        features.append(float(self._position))
        features.append(float(self._entry_price / spot - 1.0) if self._entry_price > 0 and spot > 0 else 0.0)

        obs = np.array(features, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=3.0, neginf=-3.0)
        return obs

    def _get_info(self) -> dict:
        return {
            "ticker": self.episodes[self._ep_idx]["ticker"],
            "expiration": str(self.episodes[self._ep_idx]["expiration"]),
            "step": self._step,
            "total_steps": self._ep_steps,
            "position": self._position,
            "total_reward": self._total_reward,
            "total_pnl": self._total_pnl,
            "n_trades": self._n_trades,
        }

    def step(self, action: int):
        assert action in (0, 1, 2)

        new_position = action - 1
        row = self._ep_data.iloc[self._step]
        spot = float(row["spot_price"])
        dte = int(row["days_to_expiry"])

        reward = 0.0
        trade_pnl = 0.0

        if self._position != 0 and new_position != self._position:
            slippage = self._entry_price * self.config.slippage_bps * 2 / 10000
            commission = self._entry_price * self.config.commission_bps * 2 / 10000
            direction = self._position
            trade_pnl = direction * (spot - self._entry_price) / self._entry_price * 100
            trade_pnl -= (slippage + commission) / self._entry_price * 100
            reward = trade_pnl * self.config.reward_scaling / 100
            self._total_pnl += trade_pnl
            self._n_trades += 1
            self._position = 0
            self._entry_price = 0.0

        if new_position != 0 and self._position == 0:
            self._position = new_position
            self._entry_price = spot

        if self._position != 0:
            unrealized = self._position * (spot - self._entry_price) / self._entry_price * 100
            reward += unrealized * 0.01 * self.config.reward_scaling / 100

        self._total_reward += reward
        self._step += 1

        terminated = self._step >= self._ep_steps
        truncated = False

        if terminated and self._position != 0:
            exit_price = self._expiry_close
            slippage = self._entry_price * self.config.slippage_bps * 2 / 10000
            commission = self._entry_price * self.config.commission_bps * 2 / 10000
            direction = self._position
            trade_pnl = direction * (exit_price - self._entry_price) / self._entry_price * 100
            trade_pnl -= (slippage + commission) / self._entry_price * 100
            final_reward = trade_pnl * self.config.reward_scaling / 100
            reward += final_reward
            self._total_pnl += trade_pnl
            self._n_trades += 1
            self._position = 0
            self._entry_price = 0.0

        if terminated and self._n_trades == 0:
            reward -= self.config.penalty_no_trade
            self._total_reward -= self.config.penalty_no_trade

        obs = self._get_obs() if not terminated else np.zeros(self.observation_dim, dtype=np.float32)
        info = self._get_info()

        if terminated:
            info["episode_pnl"] = self._total_pnl
            info["episode_reward"] = self._total_reward
            info["n_trades"] = self._n_trades

        return obs, float(reward), terminated, truncated, info

    def get_all_episodes_info(self) -> list[dict]:
        infos = []
        for ep in self.episodes:
            d = ep["data"]
            infos.append({
                "ticker": ep["ticker"],
                "expiration": str(ep["expiration"]),
                "n_steps": ep["n_steps"],
                "start_date": str(d.iloc[0]["quote_date"]),
                "end_date": str(d.iloc[-1]["quote_date"]),
            })
        return infos


def make_env(split: str = "train", config: EnvConfig | None = None, seed: int | None = None):
    def _init():
        env = PinningEnv(config=config, split=split)
        return env
    return _init


if __name__ == "__main__":
    config = EnvConfig(etf_only=True)
    for split in ["train", "val", "test"]:
        env = PinningEnv(config=config, split=split)
        print(f"\n{split}: {len(env.episodes)} episodes, obs_dim={env.observation_dim}")

        obs, info = env.reset(seed=42)
        print(f"  obs shape: {obs.shape}, range: [{obs.min():.3f}, {obs.max():.3f}]")
        print(f"  info: ticker={info['ticker']}, steps={info['total_steps']}")

        total_reward = 0
        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                print(f"  Episode done: pnl={info.get('episode_pnl', 0):.4f}%, "
                      f"reward={info.get('episode_reward', 0):.2f}, "
                      f"trades={info.get('n_trades', 0)}")
                break

        print(f"  Cumulative reward (partial): {total_reward:.2f}")
