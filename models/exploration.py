"""
Phase 4.6: Systematic exploration of strategy parameters, model architecture, and uncertainty.

Experiments:
  A. Strike selection: ATM vs OTM debit/credit, same-strike parity
  B. Model architecture: feature ablation (with/without GEX, OI, IV surface)
  C. Uncertainty: GEX-aware weighting, ensemble disagreement, feature-specific uncertainty
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import accuracy_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strategy.weekly_strategy import (
    load_options, _find_option_price, _round_strike, WeeklyConfig,
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

FEATURE_GROUPS = {
    "gex": ["net_gex", "net_gex_usd", "call_gex", "put_gex", "gex_flip_dist_pct",
            "net_dealer_delta", "net_vega_exposure", "net_theta_exposure",
            "total_net_gex", "total_net_dealer_delta"],
    "iv_surface": ["iv_call_10d", "iv_call_25d", "iv_call_50d", "iv_put_10d", "iv_put_25d",
                   "iv_put_50d", "risk_reversal_25d", "risk_reversal_10d", "otm_put_iv_ratio",
                   "iv_skew_25d", "iv_skew_10d", "iv_smile_width", "iv_near_term", "iv_far_term",
                   "iv_term_structure_slope", "iv_percentile_rank", "iv_rank"],
    "oi": ["oi_concentration", "oi_skew", "oi_kurtosis", "call_put_ratio", "oi_vol_ratio",
           "oi_daily_change", "oi_pct_change", "oi_3d_change", "oi_3d_pct_change",
           "oi_concentration_change", "call_put_ratio_change", "volume_daily_change"],
    "greeks": ["atm_delta", "atm_gamma", "atm_theta", "atm_vega", "atm_iv_mean", "atm_iv_std",
               "atm_spread_pct"],
    "technical": ["price_vs_sma5_pct", "price_vs_sma20_pct", "price_vs_sma50_pct",
                  "price_vs_sma200_pct", "rsi_14", "atr_pct", "atr_14", "return_5d",
                  "realized_vol_20d", "dist_to_resistance_pct", "dist_to_support_pct",
                  "macd", "macd_signal", "macd_hist", "bb_width", "bb_pct",
                  "obv", "obv_sma20", "intraday_range_pct", "return_10d", "return_20d",
                  "return_40d", "vol_ratio_5_20"],
}


def load_data():
    df = pd.read_parquet(DATA_DIR / "features_v2.parquet")
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["weekly_return_pct"] = (df["expiry_close"] - df["spot_price"]) / df["spot_price"] * 100
    return df


def prepare_features(df, feature_cols):
    for c in feature_cols:
        if c in df.columns and df[c].dtype == bool:
            df[c] = df[c].astype(int)
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

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

    class_unc = stack.std(axis=0)
    disagreement_unc = np.zeros(len(X))
    for i in range(len(X)):
        preds_i = stack[:, i, :].argmax(axis=1)
        counts = np.bincount(preds_i, minlength=3)
        disagreement_unc[i] = 1.0 - counts.max() / len(models)

    return direction, conf, unc, disagreement_unc, mean_p


# ---------------------------------------------------------------------------
# Option pricing with variable strike selection
# ---------------------------------------------------------------------------

def price_spread(opts, trade, buy_strike, sell_strike, buy_type, sell_type):
    qdate = trade["quote_date"]
    exp = trade["expiration"]

    buy_price = _find_option_price(opts, qdate, exp, buy_type, buy_strike, "ask")
    sell_price = _find_option_price(opts, qdate, exp, sell_type, sell_strike, "bid")

    if pd.isna(buy_price) or pd.isna(sell_price):
        return None

    is_debit = buy_type in ("call",) and sell_type in ("call",) and buy_strike < sell_strike
    is_debit = is_debit or (buy_type == "put" and sell_type == "put" and buy_strike > sell_strike)

    if is_debit:
        net = buy_price - sell_price
        if net <= 0:
            return None
        slippage = (buy_price + sell_price) * OPTION_SLIPPAGE_PCT / 100
        commission = 2 * COMMISSION_PER_CONTRACT
        total_cost = (net + slippage) * CONTRACT_MULTIPLIER + commission
        expiry_close = trade["expiry_close"]
        direction = trade["direction"]
        if direction == 1:
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
        expiry_close = trade["expiry_close"]
        direction = trade["direction"]
        width = abs(buy_strike - sell_strike)
        if direction == 1:
            spread_val = max(0, sell_strike - expiry_close) - max(0, buy_strike - expiry_close)
        else:
            spread_val = max(0, expiry_close - sell_strike) - max(0, expiry_close - buy_strike)
        spread_loss = spread_val * CONTRACT_MULTIPLIER
        pnl = net_credit - spread_loss
        total_cost = net_credit if net_credit > 0 else 1

    return {"pnl": pnl, "total_cost": total_cost}


# ---------------------------------------------------------------------------
# Strike strategies
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_all_experiments():
    t0 = time.time()
    df = load_data()

    train = df[df["quote_date"] <= "2024-12-31"]
    trade_base = df[(df["quote_date"] >= "2025-01-01") & (df["days_to_expiry"] >= 3) & (df["days_to_expiry"] <= 4)]
    trade_base = trade_base.groupby(["ticker", "expiration"]).last().reset_index()

    all_feat = [c for c in df.columns if c not in EXCLUDE and df[c].dtype != object and "date" not in c.lower()]

    # =========================================================================
    # B. Feature ablation: which feature groups matter?
    # =========================================================================
    print("=" * 70)
    print("B. Feature Ablation Study")
    print("=" * 70)

    ablation_configs = {
        "all_features": all_feat,
        "no_gex": [c for c in all_feat if c not in FEATURE_GROUPS["gex"]],
        "no_iv_surface": [c for c in all_feat if c not in FEATURE_GROUPS["iv_surface"]],
        "no_oi": [c for c in all_feat if c not in FEATURE_GROUPS["oi"]],
        "no_greeks": [c for c in all_feat if c not in FEATURE_GROUPS["greeks"]],
        "gex_only": FEATURE_GROUPS["gex"] + FEATURE_GROUPS["oi"] + FEATURE_GROUPS["technical"],
        "no_options": FEATURE_GROUPS["technical"],
    }

    ticker = "SPY"
    t_train = train[train["ticker"] == ticker]
    t_trade = trade_base[trade_base["ticker"] == ticker].copy()

    for name, feat_cols in ablation_configs.items():
        feat_cols = [c for c in feat_cols if c in df.columns]
        X_train = prepare_features(t_train.copy(), feat_cols)
        X_trade = prepare_features(t_trade.copy(), feat_cols)
        models = train_ensemble(X_train, t_train["weekly_return_pct"], feat_cols, n_seeds=5)
        direction, conf, unc, _, _ = predict_with_uncertainty(models, X_trade, feat_cols)

        active = direction != 0
        r = t_trade["weekly_return_pct"].values
        if active.sum() < 5:
            print(f"  {name:20s}: too few active predictions", flush=True)
            continue
        d_acc = (direction[active] > 0) == (r[active] > 0)
        d_acc = d_acc.mean() * 100
        p_acc = ((direction[active] * r[active]) > 0).mean() * 100
        avg = (direction[active] * r[active]).mean()

        mask_hc = active & (unc <= 0.04) & (conf >= 0.4)
        if mask_hc.sum() >= 3:
            d_acc_hc = ((direction[mask_hc] > 0) == (r[mask_hc] > 0)).mean() * 100
            p_acc_hc = ((direction[mask_hc] * r[mask_hc]) > 0).mean() * 100
            avg_hc = (direction[mask_hc] * r[mask_hc]).mean()
            print(f"  {name:20s}: all(n={active.sum():3d}) pnl_acc={p_acc:.0f}% avg={avg:+.2f}% | "
                  f"HC(n={mask_hc.sum():3d}) pnl_acc={p_acc_hc:.0f}% avg={avg_hc:+.2f}%", flush=True)
        else:
            print(f"  {name:20s}: all(n={active.sum():3d}) pnl_acc={p_acc:.0f}% avg={avg:+.2f}%", flush=True)

    # Run ablation for profitable tickers
    for ticker in ["GOOG", "AMZN", "NVDA"]:
        print(f"\n  --- {ticker} ablation ---", flush=True)
        t_train = train[train["ticker"] == ticker]
        t_trade = trade_base[trade_base["ticker"] == ticker].copy()
        for name, feat_cols in [("all_features", all_feat), ("no_gex", [c for c in all_feat if c not in FEATURE_GROUPS["gex"]]),
                                ("no_iv_surface", [c for c in all_feat if c not in FEATURE_GROUPS["iv_surface"]]),
                                ("no_oi", [c for c in all_feat if c not in FEATURE_GROUPS["oi"]])]:
            feat_cols = [c for c in feat_cols if c in df.columns]
            X_train = prepare_features(t_train.copy(), feat_cols)
            X_trade = prepare_features(t_trade.copy(), feat_cols)
            models = train_ensemble(X_train, t_train["weekly_return_pct"], feat_cols, n_seeds=3)
            direction, conf, unc, _, _ = predict_with_uncertainty(models, X_trade, feat_cols)
            active = direction != 0
            r = t_trade["weekly_return_pct"].values
            mask_hc = active & (unc <= 0.04) & (conf >= 0.4)
            if mask_hc.sum() >= 3:
                p_acc = ((direction[mask_hc] * r[mask_hc]) > 0).mean() * 100
                avg = (direction[mask_hc] * r[mask_hc]).mean()
                print(f"    {name:20s}: HC(n={mask_hc.sum():3d}) pnl_acc={p_acc:.0f}% avg={avg:+.2f}%", flush=True)

    # =========================================================================
    # A. Strike selection ablation (all profitable tickers, debit spread)
    # =========================================================================
    print("\n" + "=" * 70)
    print("A. Strike Selection Ablation (uncertainty-gated)")
    print("=" * 70)

    TICKER_UNC = {"SPY": (0.04, 0.40), "GOOG": (0.03, 0.45), "AMZN": (0.03, 0.45),
                  "NVDA": (0.04, 0.40), "GLD": (0.02, 0.50)}

    all_feat_df = df.copy()
    all_feat_df = prepare_features(all_feat_df, all_feat)

    ticker_models = {}
    for ticker in TICKER_UNC:
        t_train = train[train["ticker"] == ticker]
        ticker_models[ticker] = train_ensemble(
            all_feat_df[all_feat_df["ticker"] == ticker].loc[t_train.index],
            t_train["weekly_return_pct"], all_feat, n_seeds=5,
        )

    for strike_name, strike_fn in STRIKE_CONFIGS.items():
        print(f"\n  {strike_name}:", flush=True)
        for ticker, (unc_max, conf_min) in TICKER_UNC.items():
            t_trade = trade_base[trade_base["ticker"] == ticker].copy()
            if t_trade.empty or ticker not in ticker_models:
                continue

            X_trade = prepare_features(t_trade.copy(), all_feat)
            direction, conf, unc, _, _ = predict_with_uncertainty(ticker_models[ticker], X_trade, all_feat)

            opts = load_options(ticker)
            opts_idx = {}
            for (qd, exp), sub in opts.groupby(["quote_date", "expiration"]):
                opts_idx[(qd, exp)] = sub

            trades = []
            for i, (_, row) in enumerate(t_trade.iterrows()):
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

                trades.append(result["pnl"])

            if not trades:
                print(f"    {ticker:6s}: no trades", flush=True)
                continue
            pnl_arr = np.array(trades)
            sharpe = pnl_arr.mean() / pnl_arr.std() * np.sqrt(52) if pnl_arr.std() > 0 else 0
            win = (pnl_arr > 0).mean() * 100
            print(f"    {ticker:6s}: n={len(trades):3d} total={pnl_arr.sum():+8.1f} "
                  f"avg={pnl_arr.mean():+7.2f} win={win:.0f}% sharpe={sharpe:+.2f}", flush=True)

    # =========================================================================
    # C. Uncertainty improvements
    # =========================================================================
    print("\n" + "=" * 70)
    print("C. Uncertainty Improvement")
    print("=" * 70)

    for ticker in ["SPY", "GOOG", "AMZN", "NVDA"]:
        print(f"\n  {ticker}:", flush=True)
        t_train = train[train["ticker"] == ticker]
        t_trade = trade_base[trade_base["ticker"] == ticker].copy()
        X_trade = prepare_features(t_trade.copy(), all_feat)
        direction, conf, prob_unc, disagree_unc, mean_p = predict_with_uncertainty(
            ticker_models.get(ticker, train_ensemble(
                prepare_features(all_feat_df[all_feat_df["ticker"] == ticker].loc[t_train.index], all_feat),
                t_train["weekly_return_pct"], all_feat, n_seeds=5,
            )), X_trade, all_feat,
        )

        r = t_trade["weekly_return_pct"].values
        active = direction != 0

        gex_vals = t_trade["net_gex"].values if "net_gex" in t_trade.columns else np.zeros(len(t_trade))
        gex_abs = np.abs(gex_vals)

        for unc_name, unc_vals, thresh_range in [
            ("prob_std", prob_unc, [0.02, 0.03, 0.04]),
            ("disagreement", disagree_unc, [0.0, 0.2, 0.4]),
            ("gex_weighted", prob_unc * (1 + gex_abs / (gex_abs.max() + 1e-8)), [0.02, 0.03, 0.04]),
            ("inverse_gex", prob_unc / (gex_abs / (gex_abs.max() + 1e-8) + 0.5), [0.02, 0.03, 0.04]),
        ]:
            best = None
            for thresh in thresh_range:
                mask = active & (unc_vals <= thresh) & (conf >= 0.4)
                if mask.sum() < 3:
                    continue
                p_acc = ((direction[mask] * r[mask]) > 0).mean() * 100
                avg = (direction[mask] * r[mask]).mean()
                if best is None or avg > best[1]:
                    best = (thresh, avg, p_acc, mask.sum())
            if best:
                print(f"    {unc_name:15s}: best_thresh={best[0]} n={best[3]:3d} "
                      f"pnl_acc={best[2]:.0f}% avg_move={best[1]:+.2f}%", flush=True)
            else:
                print(f"    {unc_name:15s}: no valid trades", flush=True)

        print(f"    GEX correlation with correctness:", flush=True)
        correct = (direction == 0) | ((direction > 0) == (r > 0))
        for gex_split in ["positive_gex", "negative_gex"]:
            if gex_split == "positive_gex":
                gex_mask = (gex_vals > 0) & active
            else:
                gex_mask = (gex_vals <= 0) & active
            if gex_mask.sum() < 5:
                continue
            p_acc = ((direction[gex_mask] * r[gex_mask]) > 0).mean() * 100
            avg = (direction[gex_mask] * r[gex_mask]).mean()
            print(f"      {gex_split:15s}: n={gex_mask.sum():3d} pnl_acc={p_acc:.0f}% avg={avg:+.2f}%", flush=True)

    print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    run_all_experiments()
