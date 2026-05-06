"""
Phase 4.6b: Feature ablation + uncertainty improvement (no options data needed)
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

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

    disagreement_unc = np.zeros(len(X))
    for i in range(len(X)):
        preds_i = stack[:, i, :].argmax(axis=1)
        counts = np.bincount(preds_i, minlength=3)
        disagreement_unc[i] = 1.0 - counts.max() / len(models)

    direction_class_unc = np.zeros(len(X))
    for i in range(len(X)):
        if direction[i] == 1:
            direction_class_unc[i] = stack[:, i, 2].std()
        elif direction[i] == -1:
            direction_class_unc[i] = stack[:, i, 0].std()
        else:
            direction_class_unc[i] = stack[:, i, 1].std()

    return direction, conf, unc, disagreement_unc, direction_class_unc, mean_p


def run():
    t0 = time.time()
    df = load_data()

    train = df[df["quote_date"] <= "2024-12-31"]
    trade_base = df[(df["quote_date"] >= "2025-01-01") & (df["days_to_expiry"] >= 3) & (df["days_to_expiry"] <= 4)]
    trade_base = trade_base.groupby(["ticker", "expiration"]).last().reset_index()

    all_feat = [c for c in df.columns if c not in EXCLUDE and df[c].dtype != object and "date" not in c.lower()]

    ablation_configs = {
        "all_features": all_feat,
        "no_gex": [c for c in all_feat if c not in FEATURE_GROUPS["gex"]],
        "no_iv_surface": [c for c in all_feat if c not in FEATURE_GROUPS["iv_surface"]],
        "no_oi": [c for c in all_feat if c not in FEATURE_GROUPS["oi"]],
        "no_greeks": [c for c in all_feat if c not in FEATURE_GROUPS["greeks"]],
        "no_technical": [c for c in all_feat if c not in FEATURE_GROUPS["technical"]],
        "gex_oi_only": FEATURE_GROUPS["gex"] + FEATURE_GROUPS["oi"] + FEATURE_GROUPS["technical"],
        "no_options": FEATURE_GROUPS["technical"],
    }

    TICKERS = ["SPY", "GOOG", "AMZN", "NVDA", "GLD"]
    UNC_THRESHOLDS = {"SPY": (0.04, 0.40), "GOOG": (0.03, 0.45), "AMZN": (0.03, 0.45),
                      "NVDA": (0.04, 0.40), "GLD": (0.02, 0.50)}

    # =========================================================================
    print("=" * 80)
    print("B. Feature Ablation Study (per ticker, high-confidence subset)")
    print("=" * 80)

    for ticker in TICKERS:
        print(f"\n  {ticker}:", flush=True)
        t_tr = train[train["ticker"] == ticker]
        t_te = trade_base[trade_base["ticker"] == ticker].copy()
        unc_max, conf_min = UNC_THRESHOLDS.get(ticker, (0.04, 0.4))
        r = t_te["weekly_return_pct"].values

        for name, feat_cols in ablation_configs.items():
            feat_cols = [c for c in feat_cols if c in df.columns]
            X_tr = prepare_features(t_tr.copy(), feat_cols)
            X_te = prepare_features(t_te.copy(), feat_cols)
            models = train_ensemble(X_tr, t_tr["weekly_return_pct"], feat_cols, n_seeds=5)
            direction, conf, unc, _, _, _ = predict_with_uncertainty(models, X_te, feat_cols)

            active = direction != 0
            mask_hc = active & (unc <= unc_max) & (conf >= conf_min)

            results = []
            for mask, label in [(active, "all"), (mask_hc, "HC")]:
                if mask.sum() < 3:
                    results.append(f"{label}: n<3")
                    continue
                p_acc = ((direction[mask] * r[mask]) > 0).mean() * 100
                avg = (direction[mask] * r[mask]).mean()
                results.append(f"{label}(n={mask.sum():3d}): pnl={p_acc:.0f}% avg={avg:+.2f}%")
            print(f"    {name:18s}: {' | '.join(results)}", flush=True)

    # =========================================================================
    print("\n" + "=" * 80)
    print("C. Uncertainty Improvement (all features)")
    print("=" * 80)

    for ticker in TICKERS:
        print(f"\n  {ticker}:", flush=True)
        t_tr = train[train["ticker"] == ticker]
        t_te = trade_base[trade_base["ticker"] == ticker].copy()
        r = t_te["weekly_return_pct"].values
        unc_max, conf_min = UNC_THRESHOLDS.get(ticker, (0.04, 0.4))

        X_tr = prepare_features(t_tr.copy(), all_feat)
        X_te = prepare_features(t_te.copy(), all_feat)
        models = train_ensemble(X_tr, t_tr["weekly_return_pct"], all_feat, n_seeds=7)
        direction, conf, prob_unc, disagree_unc, dir_unc, mean_p = predict_with_uncertainty(models, X_te, all_feat)

        active = direction != 0

        gex_vals = t_te["net_gex"].values if "net_gex" in t_te.columns else np.zeros(len(t_te))
        gex_abs = np.abs(gex_vals)
        gex_norm = gex_abs / (gex_abs.max() + 1e-8)

        iv_rank_vals = t_te["iv_rank"].values if "iv_rank" in t_te.columns else np.full(len(t_te), 0.5)
        oi_conc_vals = t_te["oi_concentration"].values if "oi_concentration" in t_te.columns else np.full(len(t_te), 0.5)

        unc_methods = {
            "prob_std": prob_unc,
            "disagreement": disagree_unc,
            "dir_class_std": dir_unc,
            "gex_weighted": prob_unc * (1 + gex_norm),
            "inverse_gex": prob_unc / (gex_norm + 0.5),
            "oi_weighted": prob_unc * (1 + oi_conc_vals / (oi_conc_vals.max() + 1e-8)),
            "gex_oi_combined": prob_unc * (1 + 0.5 * gex_norm + 0.5 * oi_conc_vals / (oi_conc_vals.max() + 1e-8)),
            "iv_rank_adjusted": prob_unc * (1 + np.abs(iv_rank_vals - 0.5)),
        }

        for unc_name, unc_vals in unc_methods.items():
            best = None
            for thresh in np.percentile(unc_vals[active], [20, 30, 40, 50, 60]) if active.sum() > 5 else [unc_vals.max()]:
                mask = active & (unc_vals <= thresh) & (conf >= conf_min)
                if mask.sum() < 3:
                    continue
                p_acc = ((direction[mask] * r[mask]) > 0).mean() * 100
                avg = (direction[mask] * r[mask]).mean()
                sharpe = avg / np.std(direction[mask] * r[mask]) * np.sqrt(52) if mask.sum() > 3 else 0
                if best is None or avg > best[1]:
                    best = (thresh, avg, p_acc, mask.sum(), sharpe)
            if best:
                print(f"    {unc_name:20s}: best_t={best[0]:.3f} n={best[3]:3d} "
                      f"pnl_acc={best[2]:.0f}% avg={best[1]:+.2f}% sharpe={best[4]:+.2f}", flush=True)
            else:
                print(f"    {unc_name:20s}: no valid trades", flush=True)

        # GEX regime analysis
        print(f"    --- Regime analysis ---", flush=True)
        for regime_name, regime_mask in [
            ("positive_gex", (gex_vals > 0) & active),
            ("negative_gex", (gex_vals <= 0) & active),
            ("high_iv_rank", (iv_rank_vals > 0.5) & active),
            ("low_iv_rank", (iv_rank_vals <= 0.5) & active),
            ("high_oi_conc", (oi_conc_vals > np.median(oi_conc_vals)) & active),
            ("low_oi_conc", (oi_conc_vals <= np.median(oi_conc_vals)) & active),
            ("gex>0 & high_oi", (gex_vals > 0) & (oi_conc_vals > np.median(oi_conc_vals)) & active),
            ("gex>0 & low_oi", (gex_vals > 0) & (oi_conc_vals <= np.median(oi_conc_vals)) & active),
        ]:
            if regime_mask.sum() < 5:
                continue
            p_acc = ((direction[regime_mask] * r[regime_mask]) > 0).mean() * 100
            avg = (direction[regime_mask] * r[regime_mask]).mean()
            hc_mask = regime_mask & (prob_unc <= unc_max) & (conf >= conf_min)
            if hc_mask.sum() >= 3:
                p_acc_hc = ((direction[hc_mask] * r[hc_mask]) > 0).mean() * 100
                avg_hc = (direction[hc_mask] * r[hc_mask]).mean()
                print(f"      {regime_name:20s}: all(n={regime_mask.sum():3d}) pnl={p_acc:.0f}% avg={avg:+.2f}% | "
                      f"HC(n={hc_mask.sum():3d}) pnl={p_acc_hc:.0f}% avg={avg_hc:+.2f}%", flush=True)
            else:
                print(f"      {regime_name:20s}: all(n={regime_mask.sum():3d}) pnl={p_acc:.0f}% avg={avg:+.2f}%", flush=True)

    print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    run()
