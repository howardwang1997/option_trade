"""
Phase 4.6a: Strike selection ablation (needs options data)
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strategy.weekly_strategy import (
    load_options, _find_option_price, _round_strike,
    COMMISSION_PER_CONTRACT, CONTRACT_MULTIPLIER, OPTION_SLIPPAGE_PCT,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EXCLUDE = {
    "ticker", "quote_date", "expiration", "expiry_close", "expiry_high", "expiry_low",
    "spot_price", "sma_5", "sma_20", "sma_50", "sma_200", "vwap_5d",
    "max_pain_strike", "max_oi_strike", "max_call_oi_strike", "max_put_oi_strike",
    "max_net_oi_strike", "oi_mass_center", "oi_top1_strike", "max_volume_strike",
    "gex_flip_strike", "weekly_return_pct", "target_dist_pct", "pinned", "direction",
    "atm_iv",
}


def prepare_features(df, feature_cols):
    for c in feature_cols:
        if c in df.columns and df[c].dtype == bool:
            df[c] = df[c].astype(int)
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def train_ensemble(X, y, feature_cols, n_seeds=5, num_rounds=300):
    labels = np.zeros(len(y), dtype=int)
    labels[y > 0.3] = 2
    labels[y < -0.3] = 0
    labels[(y >= -0.3) & (y <= 0.3)] = 1
    models = []
    for s in range(n_seeds):
        dtrain = lgb.Dataset(X[feature_cols], label=labels)
        params = {
            "objective": "multiclass", "num_class": 3,
            "learning_rate": 0.05, "num_leaves": 63, "max_depth": 8,
            "min_child_samples": 50, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 5,
            "reg_alpha": 0.1, "reg_lambda": 0.1,
            "verbose": -1, "n_jobs": 1, "seed": s * 42,
        }
        models.append(lgb.train(params, dtrain, num_boost_round=num_rounds))
    return models


def predict_with_uncertainty(models, X, feature_cols):
    stack = np.array([m.predict(X[feature_cols]) for m in models])
    mean_p = stack.mean(axis=0)
    unc = stack.std(axis=0).mean(axis=1)
    conf = mean_p.max(axis=1)
    dir_idx = mean_p.argmax(axis=1)
    direction = np.zeros(len(X), dtype=int)
    direction[dir_idx == 2] = 1
    direction[dir_idx == 0] = -1
    return direction, conf, unc, mean_p


def price_spread(opts_sub, trade_dict, buy_strike, sell_strike, buy_type, sell_type):
    qdate = trade_dict["quote_date"]
    exp = trade_dict["expiration"]

    buy_price = _find_option_price(opts_sub, qdate, exp, buy_type, buy_strike, "ask")
    sell_price = _find_option_price(opts_sub, qdate, exp, sell_type, sell_strike, "bid")

    if pd.isna(buy_price) or pd.isna(sell_price):
        return None

    is_call_debit = buy_type == "call" and sell_type == "call" and buy_strike < sell_strike
    is_put_debit = buy_type == "put" and sell_type == "put" and buy_strike > sell_strike
    is_debit = is_call_debit or is_put_debit

    expiry_close = trade_dict["expiry_close"]
    d = trade_dict["direction"]

    if is_debit:
        net = buy_price - sell_price
        if net <= 0:
            return None
        slippage = (buy_price + sell_price) * OPTION_SLIPPAGE_PCT / 100
        commission = 2 * COMMISSION_PER_CONTRACT
        total_cost = (net + slippage) * CONTRACT_MULTIPLIER + commission
        if d == 1:
            buy_val = max(0, expiry_close - buy_strike)
            sell_val = max(0, expiry_close - sell_strike)
        else:
            buy_val = max(0, buy_strike - expiry_close)
            sell_val = max(0, sell_strike - expiry_close)
        exit_value = (buy_val - sell_val) * CONTRACT_MULTIPLIER
        pnl = exit_value - total_cost
    else:
        net = sell_price - buy_price
        if net <= 0:
            return None
        slippage = (buy_price + sell_price) * OPTION_SLIPPAGE_PCT / 100
        commission = 2 * COMMISSION_PER_CONTRACT
        net_credit = (net - slippage) * CONTRACT_MULTIPLIER - commission
        if d == 1:
            spread_val = max(0, sell_strike - expiry_close) - max(0, buy_strike - expiry_close)
        else:
            spread_val = max(0, expiry_close - sell_strike) - max(0, expiry_close - buy_strike)
        spread_loss = spread_val * CONTRACT_MULTIPLIER
        pnl = net_credit - spread_loss
        total_cost = net_credit if net_credit > 0 else 1

    return {"pnl": pnl, "total_cost": total_cost, "net": net}


STRIKE_CONFIGS = {
    "debit_atm1": lambda spot, w, d: (
        _round_strike(spot * (0.99 if d == 1 else 1.01)),
        _round_strike(spot * (0.99 if d == 1 else 1.01)) + _round_strike(spot * w / 100),
        "call" if d == 1 else "put", "call" if d == 1 else "put",
    ),
    "debit_atm3": lambda spot, w, d: (
        _round_strike(spot * (0.97 if d == 1 else 1.03)),
        _round_strike(spot * (0.97 if d == 1 else 1.03)) + _round_strike(spot * w / 100),
        "call" if d == 1 else "put", "call" if d == 1 else "put",
    ),
    "credit_otm2": lambda spot, w, d: (
        _round_strike(spot * (0.98 if d == 1 else 1.02)) - _round_strike(spot * w / 100),
        _round_strike(spot * (0.98 if d == 1 else 1.02)),
        "put" if d == 1 else "call", "put" if d == 1 else "call",
    ),
    "credit_otm4": lambda spot, w, d: (
        _round_strike(spot * (0.96 if d == 1 else 1.04)) - _round_strike(spot * w / 100),
        _round_strike(spot * (0.96 if d == 1 else 1.04)),
        "put" if d == 1 else "call", "put" if d == 1 else "call",
    ),
    "same_strike_debit": lambda spot, w, d: (
        _round_strike(spot * 0.99),
        _round_strike(spot * 0.99) + _round_strike(spot * w / 100),
        "call" if d == 1 else "put", "call" if d == 1 else "put",
    ),
    "same_strike_credit": lambda spot, w, d: (
        _round_strike(spot * 0.99) - _round_strike(spot * w / 100),
        _round_strike(spot * 0.99),
        "put" if d == 1 else "call", "put" if d == 1 else "call",
    ),
}


def run():
    t0 = time.time()
    df = pd.read_parquet(DATA_DIR / "features_v2.parquet")
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["weekly_return_pct"] = (df["expiry_close"] - df["spot_price"]) / df["spot_price"] * 100

    train = df[df["quote_date"] <= "2024-12-31"]
    trade_base = df[(df["quote_date"] >= "2025-01-01") & (df["days_to_expiry"] >= 3) & (df["days_to_expiry"] <= 4)]
    trade_base = trade_base.groupby(["ticker", "expiration"]).last().reset_index()

    all_feat = [c for c in df.columns if c not in EXCLUDE and df[c].dtype != object and "date" not in c.lower()]

    TICKER_UNC = {"SPY": (0.04, 0.40), "GOOG": (0.03, 0.45), "AMZN": (0.03, 0.45),
                  "NVDA": (0.04, 0.40), "GLD": (0.02, 0.50)}

    ticker_models = {}
    for ticker in TICKER_UNC:
        t_tr = train[train["ticker"] == ticker].copy()
        X_tr = prepare_features(t_tr, all_feat)
        ticker_models[ticker] = train_ensemble(X_tr, t_tr["weekly_return_pct"], all_feat, n_seeds=5)
        print(f"  Trained {ticker}", flush=True)

    print("=" * 80)
    print("A. Strike Selection Ablation")
    print("=" * 80)

    for strike_name, strike_fn in STRIKE_CONFIGS.items():
        print(f"\n  {strike_name}:", flush=True)
        for ticker, (unc_max, conf_min) in TICKER_UNC.items():
            t_te = trade_base[trade_base["ticker"] == ticker].copy()
            if t_te.empty or ticker not in ticker_models:
                continue

            X_te = prepare_features(t_te.copy(), all_feat)
            direction, conf, unc, _ = predict_with_uncertainty(ticker_models[ticker], X_te, all_feat)

            opts = load_options(ticker)
            opts_idx = {}
            for (qd, exp), sub in opts.groupby(["quote_date", "expiration"]):
                opts_idx[(qd, exp)] = sub

            pnls = []
            costs = []
            for i, (_, row) in enumerate(t_te.iterrows()):
                if direction[i] == 0 or unc[i] > unc_max or conf[i] < conf_min:
                    continue
                trade_dict = {
                    "quote_date": row["quote_date"], "expiration": row["expiration"],
                    "spot_price": row["spot_price"], "expiry_close": row["expiry_close"],
                    "direction": int(direction[i]),
                }
                opts_sub = opts_idx.get((row["quote_date"], row["expiration"]), pd.DataFrame())
                if opts_sub.empty:
                    continue
                spot = row["spot_price"]
                d = trade_dict["direction"]
                try:
                    buy_k, sell_k, buy_t, sell_t = strike_fn(spot, 1.5, d)
                except Exception:
                    continue
                result = price_spread(opts_sub, trade_dict, buy_k, sell_k, buy_t, sell_t)
                if result is None:
                    continue
                pnls.append(result["pnl"])
                costs.append(result["total_cost"])

            if not pnls:
                print(f"    {ticker:6s}: no trades", flush=True)
                continue
            pnl_arr = np.array(pnls)
            sharpe = pnl_arr.mean() / pnl_arr.std() * np.sqrt(52) if pnl_arr.std() > 0 else 0
            win = (pnl_arr > 0).mean() * 100
            avg_ret = pnl_arr.mean() / np.mean(costs) * 100 if costs else 0
            print(f"    {ticker:6s}: n={len(pnls):3d} total=${pnl_arr.sum():+8.0f} "
                  f"avg=${pnl_arr.mean():+7.1f} win={win:.0f}% sharpe={sharpe:+.2f} "
                  f"ret/cost={avg_ret:+.1f}%", flush=True)

    print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    run()
