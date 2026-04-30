from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

TARGET_TICKERS = ["NVDA", "GOOG", "GOOGL", "MSFT", "AMZN", "AAPL", "SPY", "QQQ", "GLD"]

MAX_DAYS_TO_EXPIRY = 5


def load_data(ticker: str):
    ohlcv = pd.read_parquet(OUTPUT_DIR / ticker / "ohlcv.parquet")
    options = pd.read_parquet(OUTPUT_DIR / ticker / "options.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    options["quote_date"] = pd.to_datetime(options["quote_date"])
    options["expiration"] = pd.to_datetime(options["expiration"])
    ohlcv = ohlcv.sort_values("date").reset_index(drop=True)
    return ohlcv, options


def compute_technical_factors_fast(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"].astype(float)

    for n in [5, 20, 50, 200]:
        df[f"sma_{n}"] = c.rolling(n, min_periods=n).mean()
        df[f"price_vs_sma{n}_pct"] = (c - df[f"sma_{n}"]) / df[f"sma_{n}"] * 100

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14, min_periods=14).mean()
    df["atr_pct"] = df["atr_14"] / c * 100

    df["return_5d"] = c.pct_change(5) * 100
    df["realized_vol_20d"] = c.pct_change().rolling(20, min_periods=20).std() * np.sqrt(252) * 100

    typical = (h + l + c) / 3
    df["vwap_5d"] = (typical * v).rolling(5, min_periods=5).sum() / v.rolling(5, min_periods=5).sum()

    df["nearest_resistance"] = c.rolling(60, min_periods=5).max()
    df["nearest_support"] = c.rolling(60, min_periods=5).min()
    df["dist_to_resistance_pct"] = (df["nearest_resistance"] - c) / c * 100
    df["dist_to_support_pct"] = (c - df["nearest_support"]) / c * 100

    return df


def analyze_ticker_fast(ticker: str) -> pd.DataFrame:
    print(f"Analyzing {ticker}...")
    ohlcv, options = load_data(ticker)

    ohlcv = compute_technical_factors_fast(ohlcv)

    trading_dates = set(ohlcv["date"].dt.strftime("%Y-%m-%d"))

    options = options[options["quote_date"] < options["expiration"]].copy()
    options["days_to_expiry"] = (options["expiration"] - options["quote_date"]).dt.days
    options = options[options["days_to_expiry"] <= MAX_DAYS_TO_EXPIRY]
    options = options[options["days_to_expiry"] >= 0]

    options["call_oi"] = np.where(options["type"] == "call", options["open_interest"], 0)
    options["put_oi"] = np.where(options["type"] == "put", options["open_interest"], 0)
    options["call_vol"] = np.where(options["type"] == "call", options["volume"].fillna(0), 0)
    options["put_vol"] = np.where(options["type"] == "put", options["volume"].fillna(0), 0)

    grp = options.groupby(["quote_date", "expiration", "strike"]).agg(
        call_oi=("call_oi", "sum"),
        put_oi=("put_oi", "sum"),
        total_oi=("open_interest", "sum"),
        total_vol=("volume", "sum"),
        call_vol=("call_vol", "sum"),
        put_vol=("put_vol", "sum"),
        avg_iv=("implied_volatility", "mean"),
        num_contracts=("open_interest", "size"),
    ).reset_index()

    grp_keys = options.groupby(["quote_date", "expiration"]).agg(
        days_to_expiry=("days_to_expiry", "first"),
    ).reset_index()[["quote_date", "expiration", "days_to_expiry"]]

    results = []
    for _, gk in grp_keys.iterrows():
        qdate = gk["quote_date"]
        exp = gk["expiration"]
        dte = gk["days_to_expiry"]

        chain = grp[(grp["quote_date"] == qdate) & (grp["expiration"] == exp)]
        if len(chain) < 2:
            continue

        total_oi = chain["total_oi"].sum()
        total_vol = chain["total_vol"].sum()
        if total_oi == 0 and total_vol == 0:
            continue

        spot_row = ohlcv[ohlcv["date"] == qdate]
        if spot_row.empty:
            continue
        spot = float(spot_row["close"].iloc[0])

        is_monthly = exp.weekday() == 4 and 15 <= exp.day <= 21

        avg_vol_20 = ohlcv[ohlcv["date"] <= qdate].tail(20)["volume"].mean()
        oi_vol_ratio = total_oi / avg_vol_20 if avg_vol_20 > 0 else np.nan

        strikes = chain["strike"].values
        call_oi_arr = chain["call_oi"].values.astype(float)
        put_oi_arr = chain["put_oi"].values.astype(float)
        total_oi_arr = chain["total_oi"].values.astype(float)
        total_vol_arr = chain["total_vol"].fillna(0).values.astype(float)

        max_oi_idx = np.argmax(total_oi_arr)
        max_oi_strike = strikes[max_oi_idx]
        oi_concentration = total_oi_arr[max_oi_idx] / total_oi_arr.sum() if total_oi_arr.sum() > 0 else 0

        max_call_oi_idx = np.argmax(call_oi_arr)
        max_call_oi_strike = strikes[max_call_oi_idx]
        max_put_oi_idx = np.argmax(put_oi_arr)
        max_put_oi_strike = strikes[max_put_oi_idx]

        net_oi = np.abs(call_oi_arr - put_oi_arr)
        max_net_idx = np.argmax(net_oi)
        max_net_oi_strike = strikes[max_net_idx]

        total_call_oi = call_oi_arr.sum()
        total_put_oi = put_oi_arr.sum()
        call_put_ratio = total_call_oi / total_put_oi if total_put_oi > 0 else np.nan

        total_oi_sum = total_oi_arr.sum()
        if total_oi_sum > 0:
            weights = total_oi_arr / total_oi_sum
            oi_mass_center = float(np.average(strikes, weights=weights))
            mean_s = oi_mass_center
            var_s = np.average((strikes - mean_s) ** 2, weights=weights)
            std_s = np.sqrt(var_s) if var_s > 0 else 1.0
            oi_skew = float(np.average(((strikes - mean_s) / std_s) ** 3, weights=weights))
            oi_kurtosis = float(np.average(((strikes - mean_s) / std_s) ** 4, weights=weights))
        else:
            oi_mass_center = np.nan
            oi_skew = 0.0
            oi_kurtosis = 0.0

        top3_idx = np.argsort(total_oi_arr)[-3:][::-1]
        oi_top1_strike = strikes[top3_idx[0]]

        max_vol_idx = np.argmax(total_vol_arr)
        max_volume_strike = strikes[max_vol_idx]

        call_intrinsic = np.maximum(0, spot - strikes) * call_oi_arr
        put_intrinsic = np.maximum(0, strikes - spot) * put_oi_arr
        total_pain = call_intrinsic + put_intrinsic
        max_pain_idx = np.argmin(total_pain)
        max_pain_strike = strikes[max_pain_idx]

        near_spot_mask = np.abs(strikes - spot) / spot < 0.05
        atm_ivs = chain.loc[near_spot_mask, "avg_iv"].dropna()
        atm_iv = float(atm_ivs.mean()) if len(atm_ivs) > 0 else np.nan

        tech_row = spot_row.iloc[0]

        last_td = ohlcv[ohlcv["date"] <= exp].iloc[-1] if not ohlcv[ohlcv["date"] <= exp].empty else None

        row = {
            "ticker": ticker,
            "expiration": exp,
            "quote_date": qdate,
            "days_to_expiry": dte,
            "is_monthly": is_monthly,
            "spot_price": spot,
            "total_oi": total_oi,
            "total_volume": total_vol,
            "oi_vol_ratio": oi_vol_ratio,
            "num_contracts": len(chain),
            "max_pain_strike": max_pain_strike,
            "max_oi_strike": max_oi_strike,
            "oi_concentration": oi_concentration,
            "max_call_oi_strike": max_call_oi_strike,
            "max_put_oi_strike": max_put_oi_strike,
            "max_net_oi_strike": max_net_oi_strike,
            "call_put_ratio": call_put_ratio,
            "oi_mass_center": oi_mass_center,
            "oi_skew": oi_skew,
            "oi_kurtosis": oi_kurtosis,
            "oi_top1_strike": oi_top1_strike,
            "max_volume_strike": max_volume_strike,
            "atm_iv": atm_iv,
            "expiry_close": float(last_td["close"]) if last_td is not None else np.nan,
            "expiry_high": float(last_td["high"]) if last_td is not None else np.nan,
            "expiry_low": float(last_td["low"]) if last_td is not None else np.nan,
        }

        for col in ["sma_5", "sma_20", "sma_50", "sma_200",
                     "price_vs_sma5_pct", "price_vs_sma20_pct", "price_vs_sma50_pct", "price_vs_sma200_pct",
                     "rsi_14", "atr_14", "atr_pct", "return_5d", "realized_vol_20d",
                     "vwap_5d", "dist_to_resistance_pct", "dist_to_support_pct"]:
            row[col] = float(tech_row[col]) if col in tech_row.index and pd.notna(tech_row[col]) else np.nan

        results.append(row)

    df = pd.DataFrame(results)
    print(f"  {ticker}: {len(df)} rows")
    return df


def run_analysis(tickers: list[str] | None = None) -> pd.DataFrame:
    if tickers is None:
        tickers = TARGET_TICKERS

    all_dfs = []
    for ticker in tickers:
        df = analyze_ticker_fast(ticker)
        all_dfs.append(df)

    result = pd.concat(all_dfs, ignore_index=True)

    out_path = Path(__file__).resolve().parent.parent / "data" / "features.parquet"
    out_path.parent.mkdir(exist_ok=True)
    result.to_parquet(out_path, index=False)
    print(f"Saved {len(result)} rows to {out_path}")
    return result


if __name__ == "__main__":
    run_analysis()
