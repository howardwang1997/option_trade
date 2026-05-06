from __future__ import annotations

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strategy.weekly_strategy import (
    WeeklyConfig, load_options, _price_debit_spread,
    ENTRY_DTE_MIN, ENTRY_DTE_MAX,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

EXCLUDE_COLS = {
    "ticker", "quote_date", "expiration", "expiry_close", "expiry_high", "expiry_low",
    "spot_price", "sma_5", "sma_20", "sma_50", "sma_200", "vwap_5d",
    "max_pain_strike", "max_oi_strike", "max_call_oi_strike", "max_put_oi_strike",
    "max_net_oi_strike", "oi_mass_center", "oi_top1_strike", "max_volume_strike",
    "gex_flip_strike", "weekly_return_pct", "target_dist_pct", "pinned", "direction",
    "atm_iv",
}

FEATURE_GROUPS = {
    "gex": {"net_gex", "net_gex_usd", "call_gex", "put_gex", "gex_flip_dist_pct",
            "net_dealer_delta", "net_vega_exposure", "net_theta_exposure",
            "total_net_gex", "total_net_dealer_delta"},
    "iv_surface": {"iv_call_10d", "iv_call_25d", "iv_call_50d", "iv_put_10d", "iv_put_25d",
                   "iv_put_50d", "risk_reversal_25d", "risk_reversal_10d", "otm_put_iv_ratio",
                   "iv_skew_25d", "iv_skew_10d", "iv_smile_width", "iv_near_term", "iv_far_term",
                   "iv_term_structure_slope", "iv_percentile_rank", "iv_rank"},
    "oi": {"oi_concentration", "oi_skew", "oi_kurtosis", "call_put_ratio", "oi_vol_ratio",
           "oi_daily_change", "oi_pct_change", "oi_3d_change", "oi_3d_pct_change",
           "oi_concentration_change", "call_put_ratio_change", "volume_daily_change"},
    "greeks": {"atm_delta", "atm_gamma", "atm_theta", "atm_vega", "atm_iv_mean", "atm_iv_std",
               "atm_spread_pct"},
    "technical": {"price_vs_sma5_pct", "price_vs_sma20_pct", "price_vs_sma50_pct",
                  "price_vs_sma200_pct", "rsi_14", "atr_pct", "atr_14", "return_5d",
                  "realized_vol_20d", "dist_to_resistance_pct", "dist_to_support_pct",
                  "macd", "macd_signal", "macd_hist", "bb_width", "bb_pct",
                  "obv", "obv_sma20", "intraday_range_pct", "return_10d", "return_20d",
                  "return_40d", "vol_ratio_5_20"},
}

TICKER_FEATURE_CONFIG = {
    "SPY":  {"exclude_groups": []},
    "GOOG": {"exclude_groups": []},
    "AMZN": {"exclude_groups": ["gex"]},
    "NVDA": {"exclude_groups": ["gex", "iv_surface"]},
    "GLD":  {"exclude_groups": []},
}

TICKER_UNC_CONFIG = {
    "SPY":  {"method": "inverse_gex",   "unc_pctile": 60, "conf_thresh": 0.40, "oi_conc_pctile": 50},
    "GOOG": {"method": "inverse_gex",   "unc_pctile": 30, "conf_thresh": 0.45, "oi_conc_pctile": 0},
    "AMZN": {"method": "prob_std",      "unc_pctile": 50, "conf_thresh": 0.45, "oi_conc_pctile": 0},
    "NVDA": {"method": "gex_weighted",  "unc_pctile": 50, "conf_thresh": 0.40, "oi_conc_pctile": 0},
    "GLD":  {"method": "dir_class_std", "unc_pctile": 40, "conf_thresh": 0.50, "oi_conc_pctile": 0},
}

TRADE_TICKERS = ["SPY", "GOOG", "AMZN", "NVDA", "GLD"]
MIN_TRAIN_YEAR = 2020
MAX_TRADE_YEAR = 2026
MIN_TRAIN_SAMPLES = 80


def _get_all_feature_cols(df):
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


def _get_ticker_features(all_features, ticker):
    config = TICKER_FEATURE_CONFIG.get(ticker, {"exclude_groups": []})
    exclude = set()
    for g in config.get("exclude_groups", []):
        exclude |= FEATURE_GROUPS.get(g, set())
    return [c for c in all_features if c not in exclude]


def load_data():
    df = pd.read_parquet(DATA_DIR / "features_v2.parquet")
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["weekly_return_pct"] = (df["expiry_close"] - df["spot_price"]) / df["spot_price"] * 100
    df["year"] = df["quote_date"].dt.year

    all_feat = _get_all_feature_cols(df)
    for c in all_feat:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    df[all_feat] = df[all_feat].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df, all_feat


class DirectionModel:
    def __init__(self, feat_cols, n_seeds=7):
        self.feat_cols = feat_cols
        self.n_seeds = n_seeds
        self.models = []

    def train(self, X, y):
        labels = np.zeros(len(y), dtype=int)
        labels[y > 0.3] = 2
        labels[y < -0.3] = 0
        labels[(y >= -0.3) & (y <= 0.3)] = 1
        self.models = []
        for seed in range(self.n_seeds):
            dtrain = lgb.Dataset(X[self.feat_cols], label=labels)
            params = {
                "objective": "multiclass", "num_class": 3,
                "learning_rate": 0.05, "num_leaves": 63, "max_depth": 8,
                "min_child_samples": 50, "feature_fraction": 0.8,
                "bagging_fraction": 0.8, "bagging_freq": 5,
                "reg_alpha": 0.1, "reg_lambda": 0.1,
                "verbose": -1, "n_jobs": 1, "seed": seed * 42,
            }
            self.models.append(lgb.train(params, dtrain, num_boost_round=300))

    def predict_full(self, X):
        stack = np.array([m.predict(X[self.feat_cols]) for m in self.models])
        mean_p = stack.mean(axis=0)
        conf = mean_p.max(axis=1)
        dir_idx = mean_p.argmax(axis=1)
        direction = np.zeros(len(X), dtype=int)
        direction[dir_idx == 2] = 1
        direction[dir_idx == 0] = -1

        prob_unc = stack.std(axis=0).mean(axis=1)
        dir_class_unc = np.zeros(len(X))
        for i in range(len(X)):
            if direction[i] == 1:
                dir_class_unc[i] = stack[:, i, 2].std()
            elif direction[i] == -1:
                dir_class_unc[i] = stack[:, i, 0].std()
            else:
                dir_class_unc[i] = stack[:, i, 1].std()

        return direction, conf, prob_unc, dir_class_unc


def compute_gated_uncertainty(prob_unc, dir_class_unc, gex_abs_norm, oi_conc_norm,
                              iv_rank, method):
    if method == "inverse_gex":
        return prob_unc / (gex_abs_norm + 0.5)
    elif method == "gex_weighted":
        return prob_unc * (1 + gex_abs_norm)
    elif method == "oi_weighted":
        return prob_unc * (1 + oi_conc_norm)
    elif method == "gex_oi_combined":
        return prob_unc * (1 + 0.5 * gex_abs_norm + 0.5 * oi_conc_norm)
    elif method == "iv_rank_adjusted":
        return prob_unc * (1 + np.abs(iv_rank - 0.5))
    elif method == "dir_class_std":
        return dir_class_unc
    return prob_unc


def run():
    t0 = time.time()
    df, all_feat = load_data()

    entry_mask = (df["days_to_expiry"] >= ENTRY_DTE_MIN) & (df["days_to_expiry"] <= ENTRY_DTE_MAX)

    print("=== Loading options data ===")
    all_opts_idx = {}
    for ticker in TRADE_TICKERS:
        opts = load_options(ticker)
        if opts.empty:
            continue
        idx = {}
        for (qd, exp), sub in opts.groupby(["quote_date", "expiration"]):
            idx[(qd, exp)] = sub
        all_opts_idx[ticker] = idx
        print(f"  {ticker}: {len(opts):,} option rows, {len(idx):,} date/exp combos")

    # =========================================================================
    # Expanding window: train on all data before year Y, trade year Y
    # =========================================================================
    print("\n=== Expanding Window Backtest (2020-2026) ===")

    all_trades = []

    for trade_year in range(2025, 2027):
        print(f"\n--- Trade Year {trade_year} ---")
        train_end = f"2024-12-31"
        trade_start = f"{trade_year}-01-01"
        trade_end = f"{trade_year}-12-31"

        train_df = df[df["quote_date"] <= train_end]
        trade_df = df[(df["quote_date"] >= trade_start) & (df["quote_date"] <= trade_end) & entry_mask]
        trade_df = trade_df.groupby(["ticker", "expiration"]).last().reset_index()
        trade_df = trade_df.sort_values(["ticker", "expiration"]).reset_index(drop=True)

        print(f"  Train: {len(train_df):,}, Trade candidates: {len(trade_df)}", flush=True)

        if trade_df.empty:
            continue

        ticker_models = {}
        ticker_feats = {}

        for ticker in TRADE_TICKERS:
            t_feat = _get_ticker_features(all_feat, ticker)
            t_train = train_df[train_df["ticker"] == ticker]

            if len(t_train) < MIN_TRAIN_SAMPLES:
                if "ALL" not in ticker_models:
                    m = DirectionModel(all_feat, n_seeds=5)
                    m.train(train_df[all_feat], train_df["weekly_return_pct"])
                    ticker_models["ALL"] = m
                    ticker_feats["ALL"] = all_feat
                ticker_models[ticker] = ticker_models["ALL"]
                ticker_feats[ticker] = all_feat
            else:
                m = DirectionModel(t_feat, n_seeds=5)
                m.train(t_train[t_feat], t_train["weekly_return_pct"])
                ticker_models[ticker] = m
                ticker_feats[ticker] = t_feat

        year_trades = []
        for ticker in TRADE_TICKERS:
            t_trade = trade_df[trade_df["ticker"] == ticker]
            if t_trade.empty or ticker not in ticker_models or ticker not in all_opts_idx:
                continue

            model = ticker_models[ticker]
            feat = ticker_feats[ticker]
            direction, conf, prob_unc, dir_class_unc = model.predict_full(t_trade[feat])

            gex_vals = t_trade["net_gex"].values if "net_gex" in t_trade.columns else np.zeros(len(t_trade))
            gex_abs = np.abs(gex_vals)
            gex_norm = gex_abs / (gex_abs.max() + 1e-8)

            oi_conc = t_trade["oi_concentration"].values if "oi_concentration" in t_trade.columns else np.full(len(t_trade), 0.5)
            oi_norm = oi_conc / (oi_conc.max() + 1e-8)

            iv_rank = t_trade["iv_rank"].values if "iv_rank" in t_trade.columns else np.full(len(t_trade), 0.5)

            unc_cfg = TICKER_UNC_CONFIG.get(ticker, {
                "method": "prob_std", "unc_pctile": 50, "conf_thresh": 0.4, "oi_conc_pctile": 0,
            })
            gated_unc = compute_gated_uncertainty(
                prob_unc, dir_class_unc, gex_norm, oi_norm, iv_rank, unc_cfg["method"],
            )

            active_mask = direction != 0
            if active_mask.sum() < 3:
                continue

            unc_thresh = np.percentile(gated_unc[active_mask], unc_cfg["unc_pctile"])
            oi_conc_pctile_thresh = unc_cfg.get("oi_conc_pctile", 0)
            oi_conc_thresh = np.percentile(oi_conc, oi_conc_pctile_thresh) if oi_conc_pctile_thresh > 0 else 0

            opts_idx = all_opts_idx[ticker]
            ticker_trades = 0

            for i, (idx, row) in enumerate(t_trade.iterrows()):
                if direction[i] == 0:
                    continue
                if gated_unc[i] > unc_thresh:
                    continue
                if conf[i] < unc_cfg["conf_thresh"]:
                    continue
                if oi_conc_thresh > 0 and oi_conc[i] < oi_conc_thresh:
                    continue

                opts_sub = opts_idx.get((row["quote_date"], row["expiration"]), pd.DataFrame())
                if opts_sub.empty:
                    continue

                trade = {
                    "quote_date": row["quote_date"], "expiration": row["expiration"],
                    "spot_price": row["spot_price"], "expiry_close": row["expiry_close"],
                    "direction": int(direction[i]),
                }
                pricing = _price_debit_spread(trade, opts_sub, WeeklyConfig(spread_width_pct=1.5))
                if pd.isna(pricing.get("pnl")):
                    continue

                year_trades.append({
                    "ticker": ticker, "year": trade_year,
                    "date": row["quote_date"],
                    "direction": int(direction[i]),
                    "confidence": float(conf[i]),
                    "gated_unc": float(gated_unc[i]),
                    "pnl": pricing["pnl"],
                    "return_pct": pricing["return_pct"],
                })
                ticker_trades += 1

            if ticker_trades > 0:
                pnls = [t["pnl"] for t in year_trades if t["ticker"] == ticker]
                a = np.array(pnls)
                s = a.mean() / a.std() * np.sqrt(52) if a.std() > 0 else 0
                print(f"  {ticker:6s}: n={len(pnls):3d} total=${a.sum():+8.0f} "
                      f"avg=${a.mean():+7.1f} win={(a>0).mean()*100:.0f}% sharpe={s:+.2f}", flush=True)

        all_trades.extend(year_trades)

        if year_trades:
            yt = np.array([t["pnl"] for t in year_trades])
            print(f"  YEAR TOTAL: n={len(yt)} total=${yt.sum():+,.0f} "
                  f"avg=${yt.mean():+7.1f} win={(yt>0).mean()*100:.0f}%", flush=True)

    # =========================================================================
    # Full portfolio summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("FULL PORTFOLIO SUMMARY (2020-2026)")
    print("=" * 80)

    if not all_trades:
        print("No trades generated.")
        return

    port = pd.DataFrame(all_trades)
    port["date"] = pd.to_datetime(port["date"])

    total = port["pnl"].sum()
    win = (port["pnl"] > 0).mean() * 100

    weekly = port.groupby("date")["pnl"].sum()
    sharpe = weekly.mean() / weekly.std() * np.sqrt(52) if weekly.std() > 0 else 0

    print(f"\n  Total trades:   {len(port)}")
    print(f"  Total PnL:      ${total:+,.0f}")
    print(f"  Win rate:        {win:.0f}%")
    print(f"  Sharpe (weekly): {sharpe:+.2f}")

    print(f"\n  Per-ticker:")
    for tk, grp in port.groupby("ticker"):
        s = grp["pnl"].mean() / grp["pnl"].std() * np.sqrt(52) if grp["pnl"].std() > 0 else 0
        print(f"    {tk:6s}: n={len(grp):4d} total=${grp['pnl'].sum():+9,.0f} "
              f"avg=${grp['pnl'].mean():+7.1f} win={(grp['pnl']>0).mean()*100:.0f}% sharpe={s:+.2f}")

    print(f"\n  Per-year:")
    for yr, grp in port.groupby("year"):
        s = grp["pnl"].mean() / grp["pnl"].std() * np.sqrt(52) if grp["pnl"].std() > 0 else 0
        wk = grp.groupby("date")["pnl"].sum()
        ws = wk.mean() / wk.std() * np.sqrt(52) if wk.std() > 0 else 0
        print(f"    {yr}: n={len(grp):3d} total=${grp['pnl'].sum():+9,.0f} "
              f"win={(grp['pnl']>0).mean()*100:.0f}% sharpe={ws:+.2f}")

    print(f"\n  Per-ticker per-year:")
    for tk in TRADE_TICKERS:
        tk_port = port[port["ticker"] == tk]
        if tk_port.empty:
            continue
        line = f"    {tk:6s}: "
        for yr in range(MIN_TRAIN_YEAR, MAX_TRADE_YEAR + 1):
            yr_port = tk_port[tk_port["year"] == yr]
            if yr_port.empty:
                line += f"{yr}:-  "
            else:
                line += f"{yr}:${yr_port['pnl'].sum():+6.0f}({yr_port['pnl'].mean():+.0f})  "
        print(line)

    # Drawdown
    weekly_sorted = weekly.sort_index()
    cum = weekly_sorted.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    max_dd = dd.min()
    print(f"\n  Max drawdown:   ${max_dd:+,.0f}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    run()
