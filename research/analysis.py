from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FEATURE_COLS = [
    "oi_concentration", "call_put_ratio", "oi_skew", "oi_kurtosis",
    "oi_vol_ratio", "total_oi", "total_volume", "num_contracts",
    "days_to_expiry", "is_monthly",
    "price_vs_sma5_pct", "price_vs_sma20_pct", "price_vs_sma50_pct", "price_vs_sma200_pct",
    "rsi_14", "atr_pct", "return_5d", "realized_vol_20d",
    "dist_to_resistance_pct", "dist_to_support_pct",
    "atm_iv",
]

CANDIDATES = [
    "max_volume_strike", "max_oi_strike", "max_call_oi_strike",
    "oi_top1_strike", "oi_mass_center",
]


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "backtest_results.parquet")
    return df[df["expiry_close"].notna() & (df["expiry_close"] > 0)].copy()


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=FEATURE_COLS).copy()
    for col in ["is_monthly"]:
        df[col] = df[col].astype(int)
    for col in ["ticker"]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix="ticker", dtype=int)
            df = pd.concat([df, dummies], axis=1)
    return df


def compute_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    best_candidate = None
    best_dist = np.inf
    for col in CANDIDATES:
        dist_col = f"dist_{col}"
        if dist_col not in df.columns:
            continue
        valid = df[dist_col].notna()
        if valid.sum() == 0:
            continue
        mean_d = df.loc[valid, dist_col].mean()
        if mean_d < best_dist:
            best_dist = mean_d
            best_candidate = col

    print(f"Best candidate overall: {best_candidate} (mean dist={best_dist:.2f}%)")

    dist_col = f"dist_{best_candidate}"
    valid = df[df[dist_col].notna()].copy()

    spot_dist = (valid["expiry_close"] - valid["spot_price"]).abs() / valid["expiry_close"] * 100
    valid["convergence"] = spot_dist - valid[dist_col]

    valid["pinned_0.5pct"] = (valid[dist_col] < 0.5).astype(int)
    valid["pinned_1pct"] = (valid[dist_col] < 1.0).astype(int)
    valid["converged"] = (valid["convergence"] > 0).astype(int)

    valid["best_candidate"] = best_candidate
    valid["target_dist"] = valid[dist_col]

    return valid


def feature_importance_classification(df: pd.DataFrame) -> dict:
    results = {}

    for label_name, label_col in [("pinned_0.5pct", "pinned_0.5pct"), ("pinned_1pct", "pinned_1pct"), ("converged", "converged")]:
        print(f"\n{'='*60}")
        print(f"Random Forest: predicting {label_name}")
        print(f"{'='*60}")

        ticker_cols = [c for c in df.columns if c.startswith("ticker_")]
        feat_cols = [c for c in FEATURE_COLS if c != "is_monthly"] + ticker_cols + ["is_monthly"]
        feat_cols = [c for c in feat_cols if c in df.columns]

        X = df[feat_cols].copy()
        y = df[label_col].copy()

        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask].values
        y = y[mask].values.astype(int)

        if len(np.unique(y)) < 2:
            print(f"  Only one class, skipping")
            continue

        print(f"  Samples: {len(y)}, Positive rate: {y.mean()*100:.1f}%")

        rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=50, random_state=42, n_jobs=-1)
        tscv = TimeSeriesSplit(n_splits=5)

        scores = cross_val_score(rf, X, y, cv=tscv, scoring="roc_auc")
        print(f"  CV AUC: {scores.mean():.3f} ± {scores.std():.3f}")

        rf.fit(X, y)
        importance = pd.DataFrame({
            "feature": feat_cols,
            "importance": rf.feature_importances_,
        }).sort_values("importance", ascending=False)

        print(f"  Top 15 features:")
        for _, row in importance.head(15).iterrows():
            print(f"    {row['feature']:30s}  {row['importance']:.4f}")

        results[label_name] = {"auc": scores.mean(), "importance": importance, "model": rf}

    return results


def feature_importance_regression(df: pd.DataFrame) -> dict:
    print(f"\n{'='*60}")
    print("Random Forest Regression: predicting target distance")
    print(f"{'='*60}")

    ticker_cols = [c for c in df.columns if c.startswith("ticker_")]
    feat_cols = [c for c in FEATURE_COLS if c != "is_monthly"] + ticker_cols + ["is_monthly"]
    feat_cols = [c for c in feat_cols if c in df.columns]

    X = df[feat_cols].copy()
    y = df["target_dist"].copy()

    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask].values
    y = y[mask].values

    print(f"  Samples: {len(y)}, Mean distance: {y.mean():.2f}%")

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=50, random_state=42, n_jobs=-1)
    tscv = TimeSeriesSplit(n_splits=5)

    scores = cross_val_score(rf, X, y, cv=tscv, scoring="neg_mean_absolute_error")
    print(f"  CV MAE: {-scores.mean():.3f}% (± {scores.std():.3f})")

    rf.fit(X, y)
    importance = pd.DataFrame({
        "feature": feat_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"  Top 15 features:")
    for _, row in importance.head(15).iterrows():
        print(f"    {row['feature']:30s}  {row['importance']:.4f}")

    return {"mae": -scores.mean(), "importance": importance, "model": rf}


def best_candidate_per_condition(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("Best candidate per condition")
    print(f"{'='*60}")

    results = []
    for ticker in sorted(df["ticker"].unique()):
        for monthly in [True, False]:
            for dte in [1, 2, 3, 4, 5]:
                subset = df[(df["ticker"] == ticker) & (df["is_monthly"] == int(monthly)) & (df["days_to_expiry"] == dte)]
                if len(subset) < 30:
                    continue

                best_col = None
                best_hit = -1
                for col in CANDIDATES:
                    dist_col = f"dist_{col}"
                    if dist_col not in subset.columns:
                        continue
                    valid = subset[dist_col].dropna()
                    if len(valid) < 30:
                        continue
                    hit_1pct = (valid < 1.0).mean() * 100
                    if hit_1pct > best_hit:
                        best_hit = hit_1pct
                        best_col = col

                if best_col:
                    results.append({
                        "ticker": ticker,
                        "is_monthly": monthly,
                        "dte": dte,
                        "n": len(subset),
                        "best_candidate": best_col,
                        "hit_1pct": best_hit,
                    })

    result_df = pd.DataFrame(results)

    for ticker in sorted(result_df["ticker"].unique()):
        t = result_df[result_df["ticker"] == ticker]
        best_row = t.loc[t["hit_1pct"].idxmax()]
        print(f"  {ticker:6s} | best: {best_row['best_candidate']:25s} hit_1pct={best_row['hit_1pct']:.1f}% "
              f"| monthly={best_row['is_monthly']} dte={best_row['dte']}d n={best_row['n']}")

    return result_df


def high_probability_scenarios(df: pd.DataFrame, min_hit_rate: float = 60.0, min_samples: int = 30) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"High probability scenarios (hit_1pct >= {min_hit_rate}%, n >= {min_samples})")
    print(f"{'='*60}")

    results = []
    for col in CANDIDATES:
        dist_col = f"dist_{col}"
        if dist_col not in df.columns:
            continue
        for ticker in sorted(df["ticker"].unique()):
            for monthly in [0, 1]:
                for dte in [1, 2, 3]:
                    subset = df[(df["ticker"] == ticker) & (df["is_monthly"] == monthly) & (df["days_to_expiry"] == dte)]
                    if len(subset) < min_samples:
                        continue
                    valid = subset[dist_col].dropna()
                    if len(valid) < min_samples:
                        continue
                    hit = (valid < 1.0).mean() * 100
                    if hit >= min_hit_rate:
                        results.append({
                            "ticker": ticker,
                            "candidate": col,
                            "is_monthly": bool(monthly),
                            "dte": dte,
                            "n": len(valid),
                            "hit_1pct": hit,
                            "mean_dist": valid.mean(),
                        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("hit_1pct", ascending=False)
        print(result_df.to_string(index=False))
    else:
        print("  No scenarios found above threshold. Lowering to 50%...")
        results = []
        for col in CANDIDATES:
            dist_col = f"dist_{col}"
            if dist_col not in df.columns:
                continue
            for ticker in sorted(df["ticker"].unique()):
                for monthly in [0, 1]:
                    for dte in [1, 2, 3]:
                        subset = df[(df["ticker"] == ticker) & (df["is_monthly"] == monthly) & (df["days_to_expiry"] == dte)]
                        if len(subset) < min_samples:
                            continue
                        valid = subset[dist_col].dropna()
                        if len(valid) < min_samples:
                            continue
                        hit = (valid < 1.0).mean() * 100
                        if hit >= 50.0:
                            results.append({
                                "ticker": ticker,
                                "candidate": col,
                                "is_monthly": bool(monthly),
                                "dte": dte,
                                "n": len(valid),
                                "hit_1pct": hit,
                                "mean_dist": valid.mean(),
                            })
        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df = result_df.sort_values("hit_1pct", ascending=False)
            print(result_df.head(30).to_string(index=False))

    return result_df


def run_analysis():
    print("Loading data...")
    df = load_data()
    print(f"Total rows: {len(df)}")

    df = compute_target_labels(df)
    df = prepare_features(df)
    print(f"After filtering: {len(df)} rows")

    cls_results = feature_importance_classification(df)
    reg_results = feature_importance_regression(df)
    condition_df = best_candidate_per_condition(df)
    scenarios_df = high_probability_scenarios(df)

    return {
        "classification": cls_results,
        "regression": reg_results,
        "conditions": condition_df,
        "scenarios": scenarios_df,
    }


if __name__ == "__main__":
    run_analysis()
