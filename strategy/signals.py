from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ETF_TICKERS = {"SPY", "QQQ", "GLD"}
STOCK_TICKERS = {"NVDA", "GOOG", "GOOGL", "MSFT", "AMZN", "AAPL"}

TARGET_CANDIDATE = "max_volume_strike"


@dataclass
class SignalConfig:
    target_candidate: str = TARGET_CANDIDATE
    etf_only: bool = True
    min_oi_concentration: float = 0.05
    max_atr_pct: float = 1.5
    min_dte: int = 3
    max_dte: int = 5
    min_direction_confidence: float = 0.0
    allowed_tickers: list[str] | None = None


def load_features(path: Path | None = None) -> pd.DataFrame:
    p = path or DATA_DIR / "features.parquet"
    df = pd.read_parquet(p)
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    return df


def compute_signal(row: pd.Series, config: SignalConfig) -> dict:
    candidate = row[config.target_candidate]
    spot = row["spot_price"]
    dte = row["days_to_expiry"]
    atr_pct = row["atr_pct"]
    oi_conc = row["oi_concentration"]

    if pd.isna(candidate) or candidate <= 0 or pd.isna(spot) or spot <= 0:
        return {"direction": 0, "confidence": 0.0, "reason": "missing_data"}

    if config.etf_only and row["ticker"] not in ETF_TICKERS:
        return {"direction": 0, "confidence": 0.0, "reason": "non_etf"}

    if dte < config.min_dte or dte > config.max_dte:
        return {"direction": 0, "confidence": 0.0, "reason": "dte_out_of_range"}

    if atr_pct > config.max_atr_pct:
        return {"direction": 0, "confidence": 0.0, "reason": "high_volatility"}

    if oi_conc < config.min_oi_concentration:
        return {"direction": 0, "confidence": 0.0, "reason": "low_oi_concentration"}

    target_dist_pct = (candidate - spot) / spot * 100

    direction = 0
    if abs(target_dist_pct) < 0.3:
        direction = 0
    elif target_dist_pct > 0:
        direction = 1
    else:
        direction = -1

    if direction == 0:
        return {"direction": 0, "confidence": 0.0, "reason": "target_near_spot"}

    confidence = _compute_confidence(row, direction, config)
    if confidence < config.min_direction_confidence:
        return {"direction": 0, "confidence": confidence, "reason": "low_confidence"}

    reason = "long_signal" if direction == 1 else "short_signal"
    return {"direction": direction, "confidence": confidence, "reason": reason}


def _compute_confidence(row: pd.Series, direction: int, config: SignalConfig) -> float:
    atr_pct = row["atr_pct"]
    oi_conc = row["oi_concentration"]
    target_dist_pct = abs(row[config.target_candidate] - row["spot_price"]) / row["spot_price"] * 100

    vol_score = max(0.0, min(1.0, (config.max_atr_pct - atr_pct) / config.max_atr_pct))
    conc_score = min(1.0, oi_conc / 0.20)
    target_score = min(1.0, target_dist_pct / 5.0)

    if direction == -1:
        direction_penalty = 0.7
    else:
        direction_penalty = 1.0

    confidence = (
        0.40 * vol_score
        + 0.35 * conc_score
        + 0.25 * target_score
    ) * direction_penalty

    return round(max(0.0, min(1.0, confidence)), 4)


def generate_signals(df: pd.DataFrame | None = None, config: SignalConfig | None = None) -> pd.DataFrame:
    if df is None:
        df = load_features()
    if config is None:
        config = SignalConfig()

    if config.allowed_tickers:
        df = df[df["ticker"].isin(config.allowed_tickers)].copy()

    signals = df.apply(lambda row: compute_signal(row, config), axis=1)
    signal_df = pd.DataFrame(signals.tolist())

    result = pd.concat([df.reset_index(drop=True), signal_df], axis=1)

    active = result[result["direction"] != 0]
    print(f"Signals generated: {len(active)} active / {len(result)} total")
    if len(active) > 0:
        print(f"  Long:  {(active['direction'] == 1).sum()}")
        print(f"  Short: {(active['direction'] == -1).sum()}")
        print(f"  Mean confidence: {active['confidence'].mean():.3f}")
        reasons = active["reason"].value_counts()
        for r, c in reasons.items():
            print(f"  {r}: {c}")

    return result


def generate_weekly_signals(df: pd.DataFrame | None = None, config: SignalConfig | None = None) -> pd.DataFrame:
    if df is None:
        df = load_features()
    if config is None:
        config = SignalConfig()

    if config.allowed_tickers:
        df = df[df["ticker"].isin(config.allowed_tickers)].copy()

    df = df.sort_values(["ticker", "expiration", "quote_date"]).reset_index(drop=True)

    results = []
    for (ticker, expiration), group in df.groupby(["ticker", "expiration"]):
        entry_rows = group[(group["days_to_expiry"] >= config.min_dte) & (group["days_to_expiry"] <= config.max_dte)]

        if entry_rows.empty:
            continue

        entry_row = entry_rows.iloc[-1]

        signal = compute_signal(entry_row, config)
        if signal["direction"] == 0:
            continue

        expiry_close = group.iloc[0]["expiry_close"] if "expiry_close" in group.columns else np.nan
        expiry_high = group.iloc[0]["expiry_high"] if "expiry_high" in group.columns else np.nan
        expiry_low = group.iloc[0]["expiry_low"] if "expiry_low" in group.columns else np.nan

        results.append({
            "ticker": ticker,
            "expiration": expiration,
            "entry_date": entry_row["quote_date"],
            "entry_price": entry_row["spot_price"],
            "entry_dte": entry_row["days_to_expiry"],
            "direction": signal["direction"],
            "confidence": signal["confidence"],
            "reason": signal["reason"],
            "target_strike": entry_row[config.target_candidate],
            "atr_pct": entry_row["atr_pct"],
            "oi_concentration": entry_row["oi_concentration"],
            "expiry_close": expiry_close,
            "expiry_high": expiry_high,
            "expiry_low": expiry_low,
        })

    result = pd.DataFrame(results)
    if not result.empty:
        result = result.sort_values(["ticker", "expiration"]).reset_index(drop=True)
        print(f"Weekly signals: {len(result)} trades")
        print(f"  Long:  {(result['direction'] == 1).sum()}")
        print(f"  Short: {(result['direction'] == -1).sum()}")
        print(f"  Mean confidence: {result['confidence'].mean():.3f}")

    return result


if __name__ == "__main__":
    configs = {
        "conservative": SignalConfig(
            etf_only=True,
            min_oi_concentration=0.05,
            max_atr_pct=1.0,
            min_dte=3,
            max_dte=5,
        ),
        "moderate": SignalConfig(
            etf_only=True,
            min_oi_concentration=0.03,
            max_atr_pct=1.5,
            min_dte=3,
            max_dte=5,
        ),
        "aggressive": SignalConfig(
            etf_only=False,
            min_oi_concentration=0.03,
            max_atr_pct=2.0,
            min_dte=2,
            max_dte=5,
        ),
    }

    for name, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"Config: {name}")
        print(f"{'='*60}")
        weekly = generate_weekly_signals(config=cfg)
        if not weekly.empty:
            out_path = DATA_DIR / f"signals_{name}.parquet"
            weekly.to_parquet(out_path, index=False)
            print(f"Saved to {out_path}")
