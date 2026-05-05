from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

TICKERS = ["SPY", "QQQ", "GLD", "AAPL", "MSFT", "AMZN", "NVDA", "GOOG", "GOOGL"]

MIN_DTE = 1
MAX_DTE = 10


def load_ohlcv(ticker: str) -> pd.DataFrame:
    p = OUTPUT_DIR / ticker / "ohlcv.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_options(ticker: str) -> pd.DataFrame:
    p = OUTPUT_DIR / ticker / "options.parquet"
    if not p.exists():
        return pd.DataFrame()
    cols = [
        "quote_date", "expiration", "type", "strike",
        "bid", "ask", "volume", "open_interest",
        "delta", "gamma", "theta", "vega", "implied_volatility",
    ]
    df = pd.read_parquet(p, columns=cols)
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["dte"] = (df["expiration"] - df["quote_date"]).dt.days
    return df


# ---------------------------------------------------------------------------
# A. GEX + Dealer Positioning
# ---------------------------------------------------------------------------

def compute_gex_features(opts: pd.DataFrame, spot: float) -> dict:
    valid = opts.dropna(subset=["gamma", "open_interest", "delta"]).copy()
    if valid.empty or spot <= 0:
        return _empty_gex()

    sign_map = {"call": 1.0, "put": -1.0}
    valid["_sign"] = valid["type"].map(sign_map).fillna(0)

    valid["_gamma_oi"] = valid["gamma"] * valid["open_interest"] * 100.0 * valid["_sign"]
    valid["_delta_oi"] = valid["delta"] * valid["open_interest"] * 100.0 * valid["_sign"]
    valid["_vega_oi"] = valid["vega"] * valid["open_interest"] * 100.0
    valid["_theta_oi"] = valid["theta"] * valid["open_interest"] * 100.0

    call_mask = valid["type"] == "call"
    put_mask = valid["type"] == "put"

    net_gex = valid["_gamma_oi"].sum()
    call_gex = valid.loc[call_mask, "gamma"].mul(valid.loc[call_mask, "open_interest"]).mul(100).sum()
    put_gex = valid.loc[put_mask, "gamma"].mul(valid.loc[put_mask, "open_interest"]).mul(-100).sum()
    net_dealer_delta = valid["_delta_oi"].sum()
    net_vega = valid["_vega_oi"].sum()
    net_theta = valid["_theta_oi"].sum()

    gex_by_strike = valid.groupby("strike")["_gamma_oi"].sum().sort_index()
    cum_gex = gex_by_strike.cumsum()
    gex_flip = np.nan
    if len(cum_gex) > 1 and cum_gex.min() < 0 < cum_gex.max():
        sign_change = np.diff(np.sign(cum_gex.values))
        flip_idx = np.where(sign_change != 0)[0]
        if len(flip_idx) > 0:
            idx = flip_idx[0]
            strikes = cum_gex.index.values
            s1, s2 = strikes[idx], strikes[idx + 1]
            g1, g2 = cum_gex.values[idx], cum_gex.values[idx + 1]
            if g2 - g1 != 0:
                gex_flip = s1 + (0 - g1) / (g2 - g1) * (s2 - s1)

    gex_flip_dist_pct = (gex_flip - spot) / spot * 100 if not np.isnan(gex_flip) and spot > 0 else np.nan
    net_gex_usd = net_gex * spot * spot / 100.0 if spot > 0 else 0

    return {
        "net_gex": net_gex,
        "net_gex_usd": net_gex_usd,
        "call_gex": call_gex,
        "put_gex": put_gex,
        "gex_flip_strike": gex_flip if not np.isnan(gex_flip) else np.nan,
        "gex_flip_dist_pct": gex_flip_dist_pct,
        "net_dealer_delta": net_dealer_delta,
        "net_vega_exposure": net_vega,
        "net_theta_exposure": net_theta,
    }


def _empty_gex() -> dict:
    return {
        "net_gex": np.nan, "net_gex_usd": np.nan,
        "call_gex": np.nan, "put_gex": np.nan,
        "gex_flip_strike": np.nan, "gex_flip_dist_pct": np.nan,
        "net_dealer_delta": np.nan, "net_vega_exposure": np.nan,
        "net_theta_exposure": np.nan,
    }


def compute_atm_greeks(opts: pd.DataFrame, spot: float) -> dict:
    if spot <= 0:
        return _empty_atm_greeks()
    near = opts[
        (opts["strike"] >= spot * 0.97) & (opts["strike"] <= spot * 1.03)
    ].dropna(subset=["delta", "gamma", "theta", "vega", "implied_volatility"])
    if near.empty:
        near = opts.dropna(subset=["delta", "gamma", "theta", "vega", "implied_volatility"])
        if near.empty:
            return _empty_atm_greeks()
        near = near.copy()
        near["_dist"] = (near["strike"] - spot).abs()
        near = near.nsmallest(20, "_dist")

    return {
        "atm_delta": near["delta"].mean(),
        "atm_gamma": near["gamma"].mean(),
        "atm_theta": near["theta"].mean(),
        "atm_vega": near["vega"].mean(),
        "atm_iv_mean": near["implied_volatility"].mean(),
        "atm_iv_std": near["implied_volatility"].std() if len(near) > 1 else 0,
        "atm_spread_pct": ((near["ask"] - near["bid"]) / ((near["ask"] + near["bid"]) / 2).replace(0, np.nan)).mean() * 100,
    }


def _empty_atm_greeks() -> dict:
    return {
        "atm_delta": np.nan, "atm_gamma": np.nan,
        "atm_theta": np.nan, "atm_vega": np.nan,
        "atm_iv_mean": np.nan, "atm_iv_std": np.nan,
        "atm_spread_pct": np.nan,
    }


# ---------------------------------------------------------------------------
# B. IV Surface Features
# ---------------------------------------------------------------------------

def compute_iv_surface(opts: pd.DataFrame, spot: float) -> dict:
    if spot <= 0:
        return _empty_iv_surface()

    valid = opts.dropna(subset=["delta", "implied_volatility"]).copy()
    valid = valid[(valid["implied_volatility"] > 0.01) & (valid["implied_volatility"] < 5.0)]

    if valid.empty:
        return _empty_iv_surface()

    def iv_at_delta(df, target_delta, opt_type):
        if opt_type == "call":
            sub = df[(df["type"] == "call") & (df["delta"] > 0)]
        else:
            sub = df[(df["type"] == "put") & (df["delta"] < 0)]
            sub = sub.copy()
            sub["delta"] = sub["delta"].abs()

        if len(sub) < 2:
            return np.nan
        sub = sub.sort_values("delta")
        return np.interp(target_delta, sub["delta"].values, sub["implied_volatility"].values)

    iv_call_10d = iv_at_delta(valid, 0.10, "call")
    iv_call_25d = iv_at_delta(valid, 0.25, "call")
    iv_call_50d = iv_at_delta(valid, 0.50, "call")
    iv_put_10d = iv_at_delta(valid, 0.10, "put")
    iv_put_25d = iv_at_delta(valid, 0.25, "put")
    iv_put_50d = iv_at_delta(valid, 0.50, "put")

    risk_reversal_25d = iv_put_25d - iv_call_25d if not (np.isnan(iv_put_25d) or np.isnan(iv_call_25d)) else np.nan
    risk_reversal_10d = iv_put_10d - iv_call_10d if not (np.isnan(iv_put_10d) or np.isnan(iv_call_10d)) else np.nan

    atm_iv = iv_call_50d if not np.isnan(iv_call_50d) else valid["implied_volatility"].median()
    otm_put_iv_ratio = iv_put_25d / atm_iv if not (np.isnan(iv_put_25d) or atm_iv == 0) else np.nan

    skew_25d = (iv_put_25d - atm_iv) * 100 if not (np.isnan(iv_put_25d) or np.isnan(atm_iv)) else np.nan
    skew_10d = (iv_put_10d - atm_iv) * 100 if not (np.isnan(iv_put_10d) or np.isnan(atm_iv)) else np.nan

    smile_width = (valid.groupby("strike")["implied_volatility"].mean().max() -
                   valid.groupby("strike")["implied_volatility"].mean().min()) * 100

    return {
        "iv_call_10d": iv_call_10d,
        "iv_call_25d": iv_call_25d,
        "iv_call_50d": iv_call_50d,
        "iv_put_10d": iv_put_10d,
        "iv_put_25d": iv_put_25d,
        "iv_put_50d": iv_put_50d,
        "risk_reversal_25d": risk_reversal_25d,
        "risk_reversal_10d": risk_reversal_10d,
        "otm_put_iv_ratio": otm_put_iv_ratio,
        "iv_skew_25d": skew_25d,
        "iv_skew_10d": skew_10d,
        "iv_smile_width": smile_width,
    }


def _empty_iv_surface() -> dict:
    keys = [
        "iv_call_10d", "iv_call_25d", "iv_call_50d",
        "iv_put_10d", "iv_put_25d", "iv_put_50d",
        "risk_reversal_25d", "risk_reversal_10d",
        "otm_put_iv_ratio", "iv_skew_25d", "iv_skew_10d",
        "iv_smile_width",
    ]
    return {k: np.nan for k in keys}


def compute_iv_term_structure(opts: pd.DataFrame) -> dict:
    near = opts[opts["dte"] <= 14].dropna(subset=["implied_volatility"])
    far = opts[(opts["dte"] > 14) & (opts["dte"] <= 60)].dropna(subset=["implied_volatility"])
    near_iv = near["implied_volatility"].median() if not near.empty else np.nan
    far_iv = far["implied_volatility"].median() if not far.empty else np.nan
    ts_slope = near_iv - far_iv if not (np.isnan(near_iv) or np.isnan(far_iv)) else np.nan

    return {
        "iv_near_term": near_iv,
        "iv_far_term": far_iv,
        "iv_term_structure_slope": ts_slope,
    }


def compute_iv_rank(opts: pd.DataFrame, iv_history: pd.Series | None = None) -> dict:
    current_iv = opts.dropna(subset=["implied_volatility"])
    current_iv = current_iv[(current_iv["implied_volatility"] > 0.01) & (current_iv["implied_volatility"] < 5.0)]
    current = current_iv["implied_volatility"].median() if not current_iv.empty else np.nan

    if iv_history is not None and len(iv_history) >= 20 and not np.isnan(current):
        pct_rank = (iv_history < current).mean() * 100
        iv_low = iv_history.quantile(0.05)
        iv_high = iv_history.quantile(0.95)
        iv_rank = (current - iv_low) / (iv_high - iv_low) * 100 if iv_high > iv_low else 50.0
    else:
        pct_rank = np.nan
        iv_rank = np.nan

    return {
        "iv_percentile_rank": pct_rank,
        "iv_rank": iv_rank,
    }


# ---------------------------------------------------------------------------
# C. OI Dynamics (time-series)
# ---------------------------------------------------------------------------

def compute_oi_dynamics(group: pd.DataFrame) -> dict:
    group = group.sort_values("quote_date").reset_index(drop=True)
    if len(group) < 2:
        return _empty_oi_dynamics()

    latest = group.iloc[-1]
    prev = group.iloc[-2]

    oi_change = latest["total_oi"] - prev["total_oi"] if "total_oi" in latest.index else np.nan
    vol_change = latest["total_volume"] - prev["total_volume"] if "total_volume" in latest.index else np.nan
    oi_pct_change = oi_change / prev["total_oi"] * 100 if "total_oi" in prev.index and prev["total_oi"] > 0 else np.nan

    if len(group) >= 4:
        oi_3d = group.iloc[-1]["total_oi"] - group.iloc[-4]["total_oi"] if "total_oi" in group.columns else np.nan
        oi_3d_pct = oi_3d / group.iloc[-4]["total_oi"] * 100 if "total_oi" in group.columns and group.iloc[-4]["total_oi"] > 0 else np.nan
    else:
        oi_3d = np.nan
        oi_3d_pct = np.nan

    conc_change = latest.get("oi_concentration", np.nan) - prev.get("oi_concentration", np.nan)
    if isinstance(conc_change, (pd.Series,)):
        conc_change = conc_change.iloc[0] if len(conc_change) > 0 else np.nan

    cpr_now = latest.get("call_put_ratio", np.nan)
    cpr_prev = prev.get("call_put_ratio", np.nan)
    cpr_change = float(cpr_now) - float(cpr_prev) if pd.notna(cpr_now) and pd.notna(cpr_prev) else np.nan

    return {
        "oi_daily_change": oi_change,
        "oi_pct_change": oi_pct_change,
        "oi_3d_change": oi_3d,
        "oi_3d_pct_change": oi_3d_pct,
        "volume_daily_change": vol_change,
        "oi_concentration_change": conc_change,
        "call_put_ratio_change": cpr_change,
    }


def _empty_oi_dynamics() -> dict:
    return {
        "oi_daily_change": np.nan, "oi_pct_change": np.nan,
        "oi_3d_change": np.nan, "oi_3d_pct_change": np.nan,
        "volume_daily_change": np.nan,
        "oi_concentration_change": np.nan,
        "call_put_ratio_change": np.nan,
    }


# ---------------------------------------------------------------------------
# D. Enhanced Technical Features
# ---------------------------------------------------------------------------

def compute_enhanced_technicals(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_width"] = (4 * std20 / sma20 * 100).replace([np.inf, -np.inf], np.nan)
    df["bb_pct"] = ((close - (sma20 - 2 * std20)) / (4 * std20)).replace([np.inf, -np.inf], np.nan)
    df["bb_pct"] = df["bb_pct"].clip(0, 1)

    obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    df["obv"] = obv
    df["obv_sma20"] = obv.rolling(20).mean()

    df["intraday_range_pct"] = ((high - low) / close * 100).replace([np.inf, -np.inf], np.nan)

    for w in [5, 10, 20, 40]:
        df[f"return_{w}d"] = close.pct_change(w) * 100

    df["vol_ratio_5_20"] = (
        df["volume"].rolling(5).mean() / df["volume"].rolling(20).mean().replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    return df


def compute_beta_to_spy(ohlcv_ticker: pd.DataFrame, ohlcv_spy: pd.DataFrame) -> pd.DataFrame:
    ret_ticker = ohlcv_ticker.set_index("date")["close"].pct_change()
    ret_spy = ohlcv_spy.set_index("date")["close"].pct_change()
    merged = pd.DataFrame({"ticker_ret": ret_ticker, "spy_ret": ret_spy}).dropna()

    beta = merged["spy_ret"].rolling(60).cov(merged["ticker_ret"]) / merged["spy_ret"].rolling(60).var()
    corr = merged["ticker_ret"].rolling(60).corr(merged["spy_ret"])

    result = pd.DataFrame({
        "date": merged.index,
        "beta_60d": beta.values,
        "corr_spy_60d": corr.values,
    }).dropna(subset=["beta_60d"])

    return result


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def extract_features_for_ticker(ticker: str, ohlcv_spy: pd.DataFrame | None = None) -> pd.DataFrame:
    print(f"  Loading {ticker}...")
    ohlcv = load_ohlcv(ticker)
    opts = load_options(ticker)
    if ohlcv.empty or opts.empty:
        print(f"  {ticker}: missing data, skipping")
        return pd.DataFrame()

    existing = pd.read_parquet(DATA_DIR / "features.parquet")
    ticker_features = existing[existing["ticker"] == ticker].copy()
    ticker_features["quote_date"] = pd.to_datetime(ticker_features["quote_date"])
    ticker_features["expiration"] = pd.to_datetime(ticker_features["expiration"])
    ticker_features = ticker_features.sort_values(["expiration", "quote_date"]).reset_index(drop=True)

    if ticker_features.empty:
        print(f"  {ticker}: no existing features, skipping")
        return pd.DataFrame()

    enhanced_tech = compute_enhanced_technicals(ohlcv)
    tech_lookup = {row["date"]: row for _, row in enhanced_tech.iterrows()}

    beta_df = pd.DataFrame()
    beta_lookup = {}
    if ohlcv_spy is not None and ticker != "SPY":
        beta_df = compute_beta_to_spy(ohlcv, ohlcv_spy)
        for _, row in beta_df.iterrows():
            beta_lookup[row["date"]] = row

    print(f"  {ticker}: building date index for options...")
    needed_dates = set(ticker_features["quote_date"].unique())
    opts_filtered = opts[opts["quote_date"].isin(needed_dates)]

    day_groups = {}
    for date, day_df in tqdm(opts_filtered.groupby("quote_date"), desc=f"  {ticker} indexing"):
        day_groups[date] = day_df

    expiry_groups = {}
    for (date, exp), sub in tqdm(
        opts_filtered.groupby(["quote_date", "expiration"]),
        desc=f"  {ticker} expiry indexing",
    ):
        expiry_groups[(date, exp)] = sub

    print(f"  {ticker}: computing features for {len(ticker_features)} rows...")

    tech_cols = [
        "macd", "macd_signal", "macd_hist", "bb_width", "bb_pct",
        "obv", "obv_sma20", "intraday_range_pct",
        "return_5d", "return_10d", "return_20d", "return_40d",
        "vol_ratio_5_20",
    ]

    results = []
    for _, row in tqdm(ticker_features.iterrows(), total=len(ticker_features), desc=f"  {ticker} features"):
        spot = row["spot_price"]
        quote_date = row["quote_date"]
        expiration = row["expiration"]

        near_expiry_opts = expiry_groups.get((quote_date, expiration), pd.DataFrame())
        all_opts_same_day = day_groups.get(quote_date, pd.DataFrame())

        result = row.to_dict()

        gex = compute_gex_features(near_expiry_opts, spot)
        result.update(gex)

        if not all_opts_same_day.empty:
            all_gex = compute_gex_features(all_opts_same_day, spot)
            result["total_net_gex"] = all_gex["net_gex"]
            result["total_net_dealer_delta"] = all_gex["net_dealer_delta"]
        else:
            result["total_net_gex"] = np.nan
            result["total_net_dealer_delta"] = np.nan

        atm = compute_atm_greeks(near_expiry_opts, spot)
        result.update(atm)

        iv_surface = compute_iv_surface(near_expiry_opts, spot)
        result.update(iv_surface)

        if not all_opts_same_day.empty:
            iv_ts = compute_iv_term_structure(all_opts_same_day)
            result.update(iv_ts)
        else:
            result.update({"iv_near_term": np.nan, "iv_far_term": np.nan, "iv_term_structure_slope": np.nan})

        tech_row = tech_lookup.get(quote_date)
        if tech_row is not None:
            for col in tech_cols:
                result[col] = tech_row.get(col, np.nan)
        else:
            for col in tech_cols:
                result[col] = np.nan

        beta_row = beta_lookup.get(quote_date)
        if beta_row is not None:
            result["beta_60d"] = beta_row["beta_60d"]
            result["corr_spy_60d"] = beta_row["corr_spy_60d"]
        else:
            result["beta_60d"] = np.nan
            result["corr_spy_60d"] = np.nan

        results.append(result)

    df = pd.DataFrame(results)

    print(f"  {ticker}: computing OI dynamics...")
    df = _add_oi_dynamics(df)
    print(f"  {ticker}: computing IV rank...")
    df = _add_iv_rank_fast(df, opts_filtered)

    print(f"  {ticker}: done, {len(df)} rows, {len(df.columns)} columns")
    return df


def _add_oi_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for col in ["oi_daily_change", "oi_pct_change", "oi_3d_change", "oi_3d_pct_change",
                "volume_daily_change", "oi_concentration_change", "call_put_ratio_change"]:
        new_cols.append(col)
        df[col] = np.nan

    for (ticker, expiration), group in df.groupby(["ticker", "expiration"]):
        idx = group.index
        n = len(group)
        if n < 2:
            continue

        total_oi = group["total_oi"].values.astype(float)
        total_vol = group["total_volume"].values.astype(float)
        oi_conc = group["oi_concentration"].values.astype(float)
        cpr = group["call_put_ratio"].values.astype(float)

        daily_change = np.full(n, np.nan)
        pct_change = np.full(n, np.nan)
        vol_change = np.full(n, np.nan)
        conc_change = np.full(n, np.nan)
        cpr_change = np.full(n, np.nan)
        change_3d = np.full(n, np.nan)
        pct_3d = np.full(n, np.nan)

        for i in range(1, n):
            if total_oi[i - 1] > 0:
                daily_change[i] = total_oi[i] - total_oi[i - 1]
                pct_change[i] = daily_change[i] / total_oi[i - 1] * 100
            vol_change[i] = total_vol[i] - total_vol[i - 1]
            conc_change[i] = oi_conc[i] - oi_conc[i - 1]
            if not (np.isnan(cpr[i]) or np.isnan(cpr[i - 1])):
                cpr_change[i] = cpr[i] - cpr[i - 1]

            if i >= 3 and total_oi[i - 3] > 0:
                change_3d[i] = total_oi[i] - total_oi[i - 3]
                pct_3d[i] = change_3d[i] / total_oi[i - 3] * 100

        df.loc[idx, "oi_daily_change"] = daily_change
        df.loc[idx, "oi_pct_change"] = pct_change
        df.loc[idx, "oi_3d_change"] = change_3d
        df.loc[idx, "oi_3d_pct_change"] = pct_3d
        df.loc[idx, "volume_daily_change"] = vol_change
        df.loc[idx, "oi_concentration_change"] = conc_change
        df.loc[idx, "call_put_ratio_change"] = cpr_change

    return df


def _add_iv_rank_fast(df: pd.DataFrame, opts_filtered: pd.DataFrame) -> pd.DataFrame:
    iv_medians = {}
    for date, day_df in opts_filtered.groupby("quote_date"):
        day_iv = day_df["implied_volatility"].dropna()
        day_iv = day_iv[(day_iv > 0.01) & (day_iv < 5.0)]
        iv_medians[date] = day_iv.median() if not day_iv.empty else np.nan

    iv_series = pd.Series(iv_medians).sort_index().dropna()

    df["iv_percentile_rank"] = np.nan
    df["iv_rank"] = np.nan

    unique_dates = sorted(df["quote_date"].unique())
    for date in unique_dates:
        current = iv_medians.get(date, np.nan)
        if np.isnan(current):
            continue
        hist = iv_series[iv_series.index < date].tail(252)
        if len(hist) < 20:
            continue
        pct = (hist < current).mean() * 100
        low = hist.quantile(0.05)
        high = hist.quantile(0.95)
        rank = (current - low) / (high - low) * 100 if high > low else 50.0
        mask = df["quote_date"] == date
        df.loc[mask, "iv_percentile_rank"] = pct
        df.loc[mask, "iv_rank"] = rank

    return df


def run_pipeline(tickers: list[str] | None = None, output_name: str = "features_v2.parquet"):
    tickers = tickers or TICKERS
    print(f"Extracting advanced features for {len(tickers)} tickers...")

    ohlcv_spy = load_ohlcv("SPY")

    all_dfs = []
    for ticker in tickers:
        df = extract_features_for_ticker(ticker, ohlcv_spy)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("No data extracted!")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["ticker", "expiration", "quote_date"]).reset_index(drop=True)

    out_path = DATA_DIR / output_name
    combined.to_parquet(out_path, index=False)
    print(f"\nSaved {len(combined)} rows x {len(combined.columns)} cols to {out_path}")
    print(f"Columns: {sorted(combined.columns.tolist())}")
    return combined


if __name__ == "__main__":
    run_pipeline()
