from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TARGET_TICKERS = ["NVDA", "GOOG", "GOOGL", "MSFT", "AMZN", "AAPL", "SPY", "QQQ", "GLD"]

CANDIDATE_STRIKES = [
    "max_pain_strike",
    "max_oi_strike",
    "max_call_oi_strike",
    "max_put_oi_strike",
    "max_net_oi_strike",
    "oi_mass_center",
    "max_volume_strike",
    "oi_top1_strike",
]


def load_features() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "features.parquet")


def compute_distances(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["expiry_close"].notna() & (df["expiry_close"] > 0)].copy()

    for col in CANDIDATE_STRIKES:
        valid = df[col].notna() & (df[col] > 0)
        df.loc[valid, f"dist_{col}"] = (
            (df.loc[valid, "expiry_close"] - df.loc[valid, col]).abs()
            / df.loc[valid, "expiry_close"] * 100
        )

    spot_dist = (df["expiry_close"] - df["spot_price"]).abs() / df["expiry_close"] * 100
    df["dist_spot_to_expiry"] = spot_dist

    for col in CANDIDATE_STRIKES:
        valid = df[col].notna() & (df[col] > 0) & df["spot_price"].notna() & (df["spot_price"] > 0)
        df.loc[valid, f"improvement_{col}"] = (
            spot_dist[valid] - df.loc[valid, f"dist_{col}"]
        )

    for col in CANDIDATE_STRIKES:
        dist_col = f"dist_{col}"
        if dist_col in df.columns:
            df[f"hit_0.5pct_{col}"] = (df[dist_col] < 0.5).astype(int)
            df[f"hit_1pct_{col}"] = (df[dist_col] < 1.0).astype(int)

    return df


def compute_baseline(df: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(42)
    baselines = []
    for ticker in df["ticker"].unique():
        t_df = df[df["ticker"] == ticker]
        prices = t_df["expiry_close"].values
        if len(prices) == 0:
            continue
        nearest_strikes = np.round(prices / 5) * 5
        random_dist = np.abs(prices - nearest_strikes) / prices * 100
        baselines.append(pd.DataFrame({
            "ticker": [ticker] * len(prices),
            "random_dist": random_dist,
        }))
    return pd.concat(baselines, ignore_index=True)


def summary_by_candidate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in CANDIDATE_STRIKES:
        dist_col = f"dist_{col}"
        if dist_col not in df.columns:
            continue
        valid = df[dist_col].dropna()
        if len(valid) == 0:
            continue
        rows.append({
            "candidate": col,
            "mean_dist_pct": valid.mean(),
            "median_dist_pct": valid.median(),
            "hit_0.5pct": (df[f"hit_0.5pct_{col}"].mean() * 100) if f"hit_0.5pct_{col}" in df.columns else np.nan,
            "hit_1pct": (df[f"hit_1pct_{col}"].mean() * 100) if f"hit_1pct_{col}" in df.columns else np.nan,
            "n": len(valid),
        })
    return pd.DataFrame(rows).sort_values("mean_dist_pct")


def summary_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in df["ticker"].unique():
        t_df = df[df["ticker"] == ticker]
        for col in CANDIDATE_STRIKES:
            dist_col = f"dist_{col}"
            if dist_col not in t_df.columns:
                continue
            valid = t_df[dist_col].dropna()
            if len(valid) < 10:
                continue
            rows.append({
                "ticker": ticker,
                "candidate": col,
                "mean_dist_pct": valid.mean(),
                "median_dist_pct": valid.median(),
                "hit_0.5pct": (t_df[f"hit_0.5pct_{col}"].mean() * 100) if f"hit_0.5pct_{col}" in t_df.columns else np.nan,
                "hit_1pct": (t_df[f"hit_1pct_{col}"].mean() * 100) if f"hit_1pct_{col}" in t_df.columns else np.nan,
                "n": len(valid),
            })
    return pd.DataFrame(rows)


def statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    baseline = compute_baseline(df)
    rows = []
    for col in CANDIDATE_STRIKES:
        dist_col = f"dist_{col}"
        if dist_col not in df.columns:
            continue
        valid = df[[dist_col, "ticker"]].dropna()
        if len(valid) < 30:
            continue

        actual = valid[dist_col].values

        baseline_vals = []
        for ticker in valid["ticker"].unique():
            b = baseline[baseline["ticker"] == ticker]["random_dist"].values
            if len(b) > 0:
                baseline_vals.extend(b[:len(valid[valid["ticker"] == ticker])])
        if len(baseline_vals) < 30:
            continue
        min_len = min(len(actual), len(baseline_vals))
        actual_sample = actual[:min_len]
        baseline_sample = np.array(baseline_vals[:min_len])

        t_stat, t_pval = stats.ttest_ind(actual_sample, baseline_sample)
        u_stat, u_pval = stats.mannwhitneyu(actual_sample, baseline_sample, alternative="less")

        pooled_std = np.sqrt((actual_sample.std() ** 2 + baseline_sample.std() ** 2) / 2)
        cohens_d = (actual_sample.mean() - baseline_sample.mean()) / pooled_std if pooled_std > 0 else 0

        rows.append({
            "candidate": col,
            "actual_mean": actual_sample.mean(),
            "baseline_mean": baseline_sample.mean(),
            "t_stat": t_stat,
            "t_pval": t_pval,
            "u_pval": u_pval,
            "cohens_d": cohens_d,
            "significant_5pct": t_pval < 0.05,
            "n": min_len,
        })
    return pd.DataFrame(rows)


def direction_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in CANDIDATE_STRIKES:
        valid = df[df[col].notna() & (df[col] > 0) & df["spot_price"].notna() & (df["spot_price"] > 0)].copy()
        if len(valid) < 10:
            continue

        price_move = valid["expiry_close"] - valid["spot_price"]
        target_move = valid[col] - valid["spot_price"]
        correct = (np.sign(price_move) == np.sign(target_move)).astype(int)

        near_zero_mask = target_move.abs() / valid["spot_price"] * 100 < 0.5
        correct.loc[near_zero_mask] = np.nan

        rows.append({
            "candidate": col,
            "direction_accuracy_pct": correct.mean() * 100,
            "n_with_direction": correct.notna().sum(),
            "n_total": len(valid),
        })
    return pd.DataFrame(rows)


def stratified_summary(df: pd.DataFrame) -> pd.DataFrame:
    col = "max_oi_strike"
    dist_col = f"dist_{col}"
    valid = df[df[dist_col].notna()].copy()

    if valid.empty:
        return pd.DataFrame()

    valid["oi_conc_group"] = pd.qcut(valid["oi_concentration"], q=3, labels=["low", "mid", "high"], duplicates="drop")
    valid["oi_vol_group"] = pd.qcut(valid["oi_vol_ratio"].clip(upper=valid["oi_vol_ratio"].quantile(0.99)),
                                     q=3, labels=["low", "mid", "high"], duplicates="drop")
    valid["dte_group"] = valid["days_to_expiry"].map({
        1: "1d", 2: "2d", 3: "3d", 4: "4d", 5: "5d"
    })

    stratifications = {
        "ticker": "ticker",
        "is_monthly": "is_monthly",
        "oi_concentration": "oi_conc_group",
        "oi_vol_ratio": "oi_vol_group",
        "days_to_expiry": "dte_group",
    }

    rows = []
    for dim_name, col_name in stratifications.items():
        if col_name not in valid.columns:
            continue
        for group_val, group_df in valid.groupby(col_name, dropna=False):
            if len(group_df) < 20:
                continue
            rows.append({
                "dimension": dim_name,
                "group": str(group_val),
                "n": len(group_df),
                "mean_dist_pct": group_df[dist_col].mean(),
                "median_dist_pct": group_df[dist_col].median(),
                "hit_0.5pct": (group_df[dist_col] < 0.5).mean() * 100,
                "hit_1pct": (group_df[dist_col] < 1.0).mean() * 100,
            })

    return pd.DataFrame(rows)


def run_backtest() -> dict:
    print("Loading features...")
    df = load_features()
    print(f"Total: {len(df)} rows")

    print("\nComputing distances...")
    df = compute_distances(df)

    print("\n=== Overall Summary by Candidate ===")
    overall = summary_by_candidate(df)
    print(overall.to_string(index=False))

    print("\n=== Statistical Tests (vs Random Baseline) ===")
    tests = statistical_tests(df)
    print(tests.to_string(index=False))

    print("\n=== Direction Accuracy ===")
    dirs = direction_accuracy(df)
    print(dirs.to_string(index=False))

    print("\n=== Best Candidate per Ticker ===")
    by_ticker = summary_by_ticker(df)
    best_per_ticker = by_ticker.loc[by_ticker.groupby("ticker")["mean_dist_pct"].idxmin()]
    print(best_per_ticker[["ticker", "candidate", "mean_dist_pct", "hit_0.5pct", "hit_1pct", "n"]].to_string(index=False))

    print("\n=== Stratified Analysis (max_oi_strike) ===")
    strat = stratified_summary(df)
    print(strat.to_string(index=False))

    out = DATA_DIR / "backtest_results.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved backtest results to {out}")

    return {
        "overall": overall,
        "tests": tests,
        "by_ticker": by_ticker,
        "direction": dirs,
        "stratified": strat,
        "data": df,
    }


if __name__ == "__main__":
    run_backtest()
