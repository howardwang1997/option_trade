from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHARTS_DIR = Path(__file__).resolve().parent.parent / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

TARGET_TICKERS = ["NVDA", "GOOG", "GOOGL", "MSFT", "AMZN", "AAPL", "SPY", "QQQ", "GLD"]


def load_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "backtest_results.parquet")


def chart_hit_rate_by_ticker(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    col = "max_volume_strike"
    dist_col = f"dist_{col}"
    d1 = df[df[dist_col].notna() & (df["days_to_expiry"] == 1)].copy()

    tickers = sorted(d1["ticker"].unique())
    hit_05 = [(d1[(d1["ticker"] == t)][dist_col] < 0.5).mean() * 100 for t in tickers]
    hit_10 = [(d1[(d1["ticker"] == t)][dist_col] < 1.0).mean() * 100 for t in tickers]

    x = np.arange(len(tickers))
    axes[0].bar(x - 0.15, hit_05, 0.3, label="< 0.5%", color="steelblue")
    axes[0].bar(x + 0.15, hit_10, 0.3, label="< 1.0%", color="coral")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tickers, rotation=45)
    axes[0].set_ylabel("Hit Rate (%)")
    axes[0].set_title("Pinning Hit Rate by Ticker\n(max_volume_strike, DTE=1)")
    axes[0].legend()
    axes[0].axhline(y=50, color="gray", linestyle="--", alpha=0.5)

    col2 = "max_oi_strike"
    dist_col2 = f"dist_{col2}"
    d2 = df[df[dist_col2].notna() & (df["days_to_expiry"] == 1)].copy()
    hit_05_2 = [(d2[(d2["ticker"] == t)][dist_col2] < 0.5).mean() * 100 for t in tickers]
    hit_10_2 = [(d2[(d2["ticker"] == t)][dist_col2] < 1.0).mean() * 100 for t in tickers]

    axes[1].bar(x - 0.15, hit_05_2, 0.3, label="< 0.5%", color="steelblue")
    axes[1].bar(x + 0.15, hit_10_2, 0.3, label="< 1.0%", color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tickers, rotation=45)
    axes[1].set_ylabel("Hit Rate (%)")
    axes[1].set_title("Pinning Hit Rate by Ticker\n(max_oi_strike, DTE=1)")
    axes[1].legend()
    axes[1].axhline(y=50, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "01_hit_rate_by_ticker.png", dpi=150)
    plt.close()
    print(f"Saved: 01_hit_rate_by_ticker.png")


def chart_dte_effect(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))

    col = "max_volume_strike"
    dist_col = f"dist_{col}"
    valid = df[df[dist_col].notna()].copy()

    for ticker in ["SPY", "QQQ", "GLD", "AAPL", "MSFT"]:
        t = valid[valid["ticker"] == ticker]
        hit_by_dte = []
        dtes = sorted(t["days_to_expiry"].unique())
        for d in dtes:
            subset = t[t["days_to_expiry"] == d][dist_col]
            hit_by_dte.append((subset < 1.0).mean() * 100)
        ax.plot(dtes, hit_by_dte, marker="o", label=ticker)

    ax.set_xlabel("Days to Expiry")
    ax.set_ylabel("Hit Rate < 1% (%)")
    ax.set_title("Pinning Hit Rate vs Days to Expiry (max_volume_strike)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "02_dte_effect.png", dpi=150)
    plt.close()
    print(f"Saved: 02_dte_effect.png")


def chart_time_stability(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5))

    col = "max_volume_strike"
    dist_col = f"dist_{col}"
    valid = df[(df[dist_col].notna()) & (df["days_to_expiry"] == 1)].copy()
    valid["year"] = valid["expiration"].dt.year

    for ticker in ["SPY", "QQQ", "GLD"]:
        t = valid[valid["ticker"] == ticker]
        yearly = t.groupby("year")[dist_col].apply(lambda x: (x < 1.0).mean() * 100)
        ax.plot(yearly.index, yearly.values, marker="o", label=ticker)

    ax.set_xlabel("Year")
    ax.set_ylabel("Hit Rate < 1% (%)")
    ax.set_title("Pinning Hit Rate Over Time (max_volume_strike, DTE=1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "03_time_stability.png", dpi=150)
    plt.close()
    print(f"Saved: 03_time_stability.png")


def chart_oi_concentration_effect(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    col = "max_volume_strike"
    dist_col = f"dist_{col}"
    valid = df[(df[dist_col].notna()) & (df["days_to_expiry"] == 1)].copy()
    valid["oi_conc_q"] = pd.qcut(valid["oi_concentration"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")

    hit_by_conc = valid.groupby("oi_conc_q")[dist_col].apply(lambda x: (x < 1.0).mean() * 100)
    mean_dist_by_conc = valid.groupby("oi_conc_q")[dist_col].mean()

    axes[0].bar(hit_by_conc.index.astype(str), hit_by_conc.values, color="steelblue")
    axes[0].set_ylabel("Hit Rate < 1% (%)")
    axes[0].set_title("Hit Rate vs OI Concentration Quintile")

    axes[1].bar(mean_dist_by_conc.index.astype(str), mean_dist_by_conc.values, color="coral")
    axes[1].set_ylabel("Mean Distance (%)")
    axes[1].set_title("Mean Distance vs OI Concentration Quintile")

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "04_oi_concentration.png", dpi=150)
    plt.close()
    print(f"Saved: 04_oi_concentration.png")


def chart_feature_importance(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))

    from sklearn.ensemble import RandomForestClassifier

    feat_cols = [
        "oi_concentration", "call_put_ratio", "oi_skew",
        "oi_vol_ratio", "total_oi", "total_volume",
        "days_to_expiry", "is_monthly",
        "price_vs_sma20_pct", "price_vs_sma50_pct",
        "rsi_14", "atr_pct", "realized_vol_20d",
        "atm_iv", "dist_to_resistance_pct", "dist_to_support_pct",
    ]

    col = "max_volume_strike"
    dist_col = f"dist_{col}"
    valid = df[df[dist_col].notna()].dropna(subset=feat_cols).copy()
    valid["pinned"] = (valid[dist_col] < 1.0).astype(int)

    X = valid[feat_cols].values
    y = valid["pinned"].values

    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importance = pd.DataFrame({
        "feature": feat_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=True)

    ax.barh(importance["feature"], importance["importance"], color="steelblue")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest Feature Importance\n(Predicting pinning within 1% of max_volume_strike)")

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "05_feature_importance.png", dpi=150)
    plt.close()
    print(f"Saved: 05_feature_importance.png")


def chart_example_pinning(df: pd.DataFrame):
    from research.option_analysis import load_data as load_raw

    ticker = "SPY"
    ohlcv, options = load_raw(ticker)

    target_exp = pd.Timestamp("2025-03-21")

    chain = options[options["expiration"] == target_exp]
    qdates = sorted(chain["quote_date"].unique())
    if len(qdates) == 0:
        print(f"No data for {ticker} exp={target_exp}")
        return

    last_qd = qdates[-1]
    last_chain = chain[chain["quote_date"] == last_qd]

    oi_by_strike = last_chain.groupby(["strike", "type"])["open_interest"].sum().unstack(fill_value=0)
    vol_by_strike = last_chain.groupby(["strike", "type"])["volume"].sum().unstack(fill_value=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]})

    ax1.bar(oi_by_strike.index, oi_by_strike.get("call", 0), width=1.5, alpha=0.6, label="Call OI", color="green")
    ax1.bar(oi_by_strike.index, -oi_by_strike.get("put", 0), width=1.5, alpha=0.6, label="Put OI", color="red")

    exp_close = df[(df["ticker"] == ticker) & (df["expiration"] == target_exp)]["expiry_close"].iloc[0]
    spot = df[(df["ticker"] == ticker) & (df["expiration"] == target_exp) & (df["days_to_expiry"] == 1)]["spot_price"].values
    if len(spot) > 0:
        ax1.axvline(x=spot[0], color="blue", linewidth=2, linestyle="-", label=f"T-1 Close: {spot[0]:.1f}")
    ax1.axvline(x=exp_close, color="black", linewidth=2, linestyle="--", label=f"Expiry Close: {exp_close:.1f}")

    max_vol_strike = vol_by_strike.sum(axis=1).idxmax()
    ax1.axvline(x=max_vol_strike, color="orange", linewidth=2, linestyle=":", label=f"Max Vol Strike: {max_vol_strike:.0f}")

    ax1.set_xlabel("Strike")
    ax1.set_ylabel("Open Interest")
    ax1.set_title(f"{ticker} Option Chain before Expiration {target_exp.strftime('%Y-%m-%d')}\n(quote_date: {last_qd.strftime('%Y-%m-%d')})")
    ax1.legend()

    price_range = ohlcv[(ohlcv["date"] >= target_exp - pd.Timedelta(days=10)) & (ohlcv["date"] <= target_exp)]
    ax2.plot(price_range["date"], price_range["close"], marker="o", color="blue", linewidth=2)
    ax2.axhline(y=max_vol_strike, color="orange", linestyle=":", label=f"Max Vol Strike: {max_vol_strike:.0f}")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")
    ax2.set_title(f"{ticker} Price Action near Expiration")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "06_example_pinning.png", dpi=150)
    plt.close()
    print(f"Saved: 06_example_pinning.png")


def run_visualize():
    print("Loading data...")
    df = load_data()

    chart_hit_rate_by_ticker(df)
    chart_dte_effect(df)
    chart_time_stability(df)
    chart_oi_concentration_effect(df)
    chart_feature_importance(df)
    chart_example_pinning(df)

    print(f"\nAll charts saved to {CHARTS_DIR}/")


if __name__ == "__main__":
    run_visualize()
