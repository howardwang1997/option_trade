from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, accuracy_score
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strategy.weekly_strategy import (
    WeeklyConfig, load_features_v2, load_options, clear_options_cache,
    _price_debit_spread, _price_credit_spread, _price_iron_condor,
    _round_strike, _find_option_price, ENTRY_DTE_MIN, ENTRY_DTE_MAX,
    COMMISSION_PER_CONTRACT, CONTRACT_MULTIPLIER, OPTION_SLIPPAGE_PCT,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

EXCLUDE_COLS = {
    "ticker", "quote_date", "expiration", "expiry_close", "expiry_high", "expiry_low",
    "spot_price", "sma_5", "sma_20", "sma_50", "sma_200", "vwap_5d",
    "max_pain_strike", "max_oi_strike", "max_call_oi_strike", "max_put_oi_strike",
    "max_net_oi_strike", "oi_mass_center", "oi_top1_strike", "max_volume_strike",
    "gex_flip_strike", "weekly_return_pct", "target_dist_pct", "pinned", "direction",
}


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in EXCLUDE_COLS:
            continue
        if df[c].dtype == object or df[c].dtype.name == "category":
            continue
        if "date" in c.lower() or "expiry" in c.lower():
            continue
        cols.append(c)
    return cols


def load_data():
    df = pd.read_parquet(DATA_DIR / "features_v2.parquet")
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["weekly_return_pct"] = (df["expiry_close"] - df["spot_price"]) / df["spot_price"] * 100

    feat_cols = _get_feature_cols(df)

    for c in feat_cols:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df, feat_cols


class DirectionModel:
    def __init__(self, feat_cols: list[str], n_classes: int = 3):
        self.feat_cols = feat_cols
        self.n_classes = n_classes
        self.models = []

    def _prepare_target(self, y: pd.Series) -> np.ndarray:
        labels = np.zeros(len(y), dtype=int)
        labels[y > 0.3] = 2
        labels[y < -0.3] = 0
        labels[(y >= -0.3) & (y <= 0.3)] = 1
        return labels

    def train(self, X: pd.DataFrame, y: pd.Series, n_seeds: int = 5, verbose: bool = True):
        labels = self._prepare_target(y)
        self.models = []

        for seed in range(n_seeds):
            dtrain = lgb.Dataset(X[self.feat_cols], label=labels)
            params = {
                "objective": "multiclass",
                "num_class": self.n_classes,
                "metric": "multi_logloss",
                "learning_rate": 0.05,
                "num_leaves": 63,
                "max_depth": 8,
                "min_child_samples": 50,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "verbose": -1,
                "n_jobs": 1,
                "seed": seed * 42,
            }
            model = lgb.train(params, dtrain, num_boost_round=300)
            self.models.append(model)

        if verbose:
            probs = self.predict_proba(X)
            pred_labels = probs.argmax(axis=1)
            acc = accuracy_score(labels, pred_labels)
            class_dist = pd.Series(labels).value_counts().sort_index()
            print(f"  Train accuracy: {acc:.3f}")
            print(f"  Class dist: down={class_dist.get(0, 0)}, flat={class_dist.get(1, 0)}, up={class_dist.get(2, 0)}")

            importance = pd.DataFrame({
                "feature": self.feat_cols,
                "importance": np.mean([m.feature_importance("gain") for m in self.models], axis=0),
            }).sort_values("importance", ascending=False)
            print("  Top 10 features:")
            for _, row in importance.head(10).iterrows():
                print(f"    {row['feature']:30s}  {row['importance']:.1f}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        all_probs = []
        for model in self.models:
            all_probs.append(model.predict(X[self.feat_cols]))
        return np.mean(all_probs, axis=0)

    def predict_with_uncertainty(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_probs = []
        for model in self.models:
            all_probs.append(model.predict(X[self.feat_cols]))
        probs_stack = np.array(all_probs)
        mean_probs = probs_stack.mean(axis=0)
        uncertainty = probs_stack.std(axis=0).mean(axis=1)

        pred_classes = mean_probs.argmax(axis=1)
        confidence = mean_probs.max(axis=1)

        direction = np.zeros(len(X), dtype=int)
        direction[pred_classes == 2] = 1
        direction[pred_classes == 0] = -1

        return direction, confidence, uncertainty


def run_uncertainty_strategy(verbose: bool = True):
    df, feat_cols = load_data()

    train_mask = df["quote_date"] <= "2024-12-31"
    trade_mask = (df["quote_date"] >= "2025-01-01") & (df["quote_date"] <= "2026-12-31")
    entry_mask = (df["days_to_expiry"] >= ENTRY_DTE_MIN) & (df["days_to_expiry"] <= ENTRY_DTE_MAX)

    train_df = df[train_mask].copy()
    trade_df = df[trade_mask & entry_mask].copy()
    trade_df = trade_df.groupby(["ticker", "expiration"]).last().reset_index()
    trade_df = trade_df.sort_values(["ticker", "expiration"]).reset_index(drop=True)

    print(f"Train: {len(train_df)} rows, Trade candidates: {len(trade_df)}")

    if verbose:
        print("\n=== Training per-ticker direction models ===")

    ticker_models = {}
    for ticker in sorted(trade_df["ticker"].unique()):
        t_train = train_df[train_df["ticker"] == ticker]
        if len(t_train) < 100:
            print(f"  {ticker}: insufficient train data ({len(t_train)}), using all-ticker model")
            if "ALL" not in ticker_models:
                print(f"\n  Training ALL-ticker model...")
                m = DirectionModel(feat_cols)
                m.train(train_df[feat_cols], train_df["weekly_return_pct"], n_seeds=5, verbose=True)
                ticker_models["ALL"] = m
            ticker_models[ticker] = ticker_models["ALL"]
        else:
            print(f"\n  Training {ticker} ({len(t_train)} samples)...")
            m = DirectionModel(feat_cols)
            m.train(t_train[feat_cols], t_train["weekly_return_pct"], n_seeds=5, verbose=True)
            ticker_models[ticker] = m

    print("\n=== Generating predictions with uncertainty ===")
    for ticker in sorted(trade_df["ticker"].unique()):
        t_trade = trade_df[trade_df["ticker"] == ticker].copy()
        if t_trade.empty:
            continue
        model = ticker_models[ticker]
        direction, confidence, uncertainty = model.predict_with_uncertainty(t_trade)
        trade_df.loc[t_trade.index, "pred_direction"] = direction
        trade_df.loc[t_trade.index, "pred_confidence"] = confidence
        trade_df.loc[t_trade.index, "pred_uncertainty"] = uncertainty

    trade_df["pred_direction"] = trade_df["pred_direction"].astype(int)
    trade_df["actual_direction"] = 0
    trade_df.loc[trade_df["weekly_return_pct"] > 0.3, "actual_direction"] = 1
    trade_df.loc[trade_df["weekly_return_pct"] < -0.3, "actual_direction"] = -1

    print("\n=== Uncertainty analysis ===")
    for ticker in sorted(trade_df["ticker"].unique()):
        t = trade_df[trade_df["ticker"] == ticker]
        if t.empty:
            continue

        unc_bins = pd.qcut(t["pred_uncertainty"], q=4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"], duplicates="drop")
        t = t.copy()
        t["unc_bin"] = unc_bins

        print(f"\n  {ticker}:")
        for bin_name in ["Q1(low)", "Q2", "Q3", "Q4(high)"]:
            sub = t[t["unc_bin"] == bin_name]
            if sub.empty:
                continue
            active = sub[sub["pred_direction"] != 0]
            if active.empty:
                print(f"    {bin_name}: n={len(sub)}, no active predictions")
                continue
            correct = (active["pred_direction"] == active["actual_direction"]).mean() * 100
            same_sign = ((active["pred_direction"] * active["weekly_return_pct"]) > 0).mean() * 100
            mean_unc = active["pred_uncertainty"].mean()
            mean_conf = active["pred_confidence"].mean()
            print(f"    {bin_name}: n={len(sub)}, active={len(active)}, "
                  f"dir_acc={correct:.0f}%, pnl_acc={same_sign:.0f}%, "
                  f"unc={mean_unc:.4f}, conf={mean_conf:.2f}")

    print("\n=== Backtest with uncertainty gating (debit spread) ===")

    all_opts_idx = {}
    for ticker in sorted(trade_df["ticker"].unique()):
        opts = load_options(ticker)
        if opts.empty:
            continue
        print(f"  Indexing {ticker}...")
        idx = {}
        for (qd, exp), sub in opts.groupby(["quote_date", "expiration"]):
            idx[(qd, exp)] = sub
        all_opts_idx[ticker] = idx

    for unc_thresh in [0.02, 0.03, 0.05]:
        for conf_thresh in [0.4, 0.5, 0.6]:
            print(f"\n--- unc<={unc_thresh}, conf>={conf_thresh} ---")
            for ticker in sorted(trade_df["ticker"].unique()):
                t = trade_df[trade_df["ticker"] == ticker]
                if t.empty or ticker not in all_opts_idx:
                    continue

                opts_idx = all_opts_idx[ticker]
                trades = []
                for _, row in t.iterrows():
                    if row["pred_direction"] == 0:
                        continue
                    if row["pred_uncertainty"] > unc_thresh:
                        continue
                    if row["pred_confidence"] < conf_thresh:
                        continue

                    trade = {
                        "ticker": ticker,
                        "expiration": row["expiration"],
                        "quote_date": row["quote_date"],
                        "spot_price": row["spot_price"],
                        "expiry_close": row["expiry_close"],
                        "direction": int(row["pred_direction"]),
                        "confidence": row["pred_confidence"],
                        "uncertainty": row["pred_uncertainty"],
                    }

                    opts_sub = opts_idx.get((row["quote_date"], row["expiration"]), pd.DataFrame())
                    if opts_sub.empty:
                        continue

                    pricing = _price_debit_spread(trade, opts_sub, WeeklyConfig(spread_width_pct=1.5))
                    if pd.isna(pricing.get("pnl")):
                        continue

                    trade.update(pricing)
                    trade["position_pnl"] = trade["pnl"] * row["pred_confidence"]
                    trades.append(trade)

                if not trades:
                    print(f"  {ticker:6s}: no trades")
                    continue

                trades_df = pd.DataFrame(trades)
                pnl = trades_df["position_pnl"]
                sharpe = pnl.mean() / pnl.std() * np.sqrt(52) if pnl.std() > 0 else 0
                win = len(pnl[pnl > 0]) / len(pnl) * 100
                total = pnl.sum()
                avg_ret = trades_df["return_pct"].mean()
                print(f"  {ticker:6s}: n={len(trades):3d}  avg_ret={avg_ret:+7.2f}%  win={win:5.1f}%  "
                      f"sharpe={sharpe:+.2f}  total_pnl={total:+8.1f}")


if __name__ == "__main__":
    results = run_uncertainty_strategy()
