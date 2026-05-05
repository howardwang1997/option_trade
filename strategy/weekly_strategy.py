from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

ETF_TICKERS = {"SPY", "QQQ", "GLD"}

ENTRY_DTE_MIN = 3
ENTRY_DTE_MAX = 4

COMMISSION_PER_CONTRACT = 0.65
CONTRACT_MULTIPLIER = 100
OPTION_SLIPPAGE_PCT = 0.5


@dataclass
class WeeklyConfig:
    strategy_type: Literal["debit_spread", "credit_spread", "iron_condor", "auto"] = "auto"
    spread_width_pct: float = 1.5
    min_confidence: float = 0.1
    direction_threshold: float = 0.2
    use_any_model: bool = True
    position_pct: float = 1.0
    tickers: list[str] | None = None
    trade_start: str | None = None
    trade_end: str | None = None
    train_end: str = "2019-12-31"
    val_end: str = "2022-12-31"


def load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "predictions.parquet")
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    return df


def load_features_v2() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "features_v2.parquet")
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    return df


_options_cache: dict[str, pd.DataFrame] = {}


def load_options(ticker: str) -> pd.DataFrame:
    if ticker in _options_cache:
        return _options_cache[ticker]
    p = OUTPUT_DIR / ticker / "options.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p, columns=[
        "quote_date", "expiration", "type", "strike",
        "bid", "ask", "volume", "open_interest",
        "delta", "implied_volatility",
    ])
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    _options_cache[ticker] = df
    return df


def clear_options_cache():
    _options_cache.clear()


def _find_option_price(options: pd.DataFrame, quote_date, expiration,
                       opt_type: str, strike: float, side: str = "mid") -> float:
    match = options[
        (options["quote_date"] == quote_date)
        & (options["expiration"] == expiration)
        & (options["type"] == opt_type)
        & (options["strike"] == strike)
    ]
    if match.empty:
        strikes = options[
            (options["quote_date"] == quote_date)
            & (options["expiration"] == expiration)
            & (options["type"] == opt_type)
        ]["strike"]
        if strikes.empty:
            return np.nan
        nearest_idx = (strikes - strike).abs().idxmin()
        match = options.loc[[nearest_idx]]

    row = match.iloc[0]
    bid = float(row["bid"]) if pd.notna(row["bid"]) else 0.0
    ask = float(row["ask"]) if pd.notna(row["ask"]) else 0.0

    if side == "bid":
        return bid
    elif side == "ask":
        return ask
    return (bid + ask) / 2 if bid + ask > 0 else np.nan


def _round_strike(price: float, tick_size: float = 0.5) -> float:
    return round(price / tick_size) * tick_size


def _select_strategy(row: pd.Series, config: WeeklyConfig) -> str:
    if config.strategy_type != "auto":
        return config.strategy_type

    direction = row.get("ensemble_direction", 0)
    net_gex = row.get("net_gex", 0)
    iv_rank = row.get("iv_rank", 50)

    if direction == 0:
        if net_gex > 0 and iv_rank > 30:
            return "iron_condor"
        return "flat"

    if iv_rank > 70:
        if direction == 1:
            return "credit_spread"
        else:
            return "credit_spread"

    if net_gex > 0:
        return "iron_condor" if abs(row.get("ensemble_return_pred", 0)) < 1.0 else "debit_spread"

    return "debit_spread"


def _price_debit_spread(trade: dict, options: pd.DataFrame, config: WeeklyConfig) -> dict:
    spot = trade["spot_price"]
    direction = trade["direction"]
    exp = trade["expiration"]
    qdate = trade["quote_date"]

    width = _round_strike(spot * config.spread_width_pct / 100)

    if direction == 1:
        buy_strike = _round_strike(spot * 0.99)
        sell_strike = buy_strike + width
        buy_type, sell_type = "call", "call"
    else:
        sell_strike = _round_strike(spot * 0.99)
        buy_strike = sell_strike + width
        buy_type, sell_type = "put", "put"

    buy_price = _find_option_price(options, qdate, exp, buy_type, buy_strike, "ask")
    sell_price = _find_option_price(options, qdate, exp, sell_type, sell_strike, "bid")

    if pd.isna(buy_price) or pd.isna(sell_price):
        return {"pnl": np.nan, "return_pct": np.nan, "reason": "no_option_data"}

    debit = buy_price - sell_price
    if debit <= 0:
        return {"pnl": np.nan, "return_pct": np.nan, "reason": "negative_debit"}

    slippage = (buy_price + sell_price) * OPTION_SLIPPAGE_PCT / 100
    commission = 2 * COMMISSION_PER_CONTRACT
    total_cost = (debit + slippage) * CONTRACT_MULTIPLIER + commission

    expiry_close = trade["expiry_close"]
    if direction == 1:
        buy_val = max(0, expiry_close - buy_strike)
        sell_val = max(0, expiry_close - sell_strike)
    else:
        buy_val = max(0, buy_strike - expiry_close)
        sell_val = max(0, sell_strike - expiry_close)

    exit_value = (buy_val - sell_val) * CONTRACT_MULTIPLIER
    pnl = exit_value - total_cost
    return_pct = pnl / total_cost * 100 if total_cost > 0 else 0

    return {
        "pnl": round(pnl, 2),
        "return_pct": round(return_pct, 2),
        "total_cost": round(total_cost, 2),
        "buy_strike": buy_strike,
        "sell_strike": sell_strike,
        "buy_type": buy_type,
        "debit": round(debit, 4),
        "reason": "ok",
    }


def _price_credit_spread(trade: dict, options: pd.DataFrame, config: WeeklyConfig) -> dict:
    spot = trade["spot_price"]
    direction = trade["direction"]
    exp = trade["expiration"]
    qdate = trade["quote_date"]

    width = _round_strike(spot * config.spread_width_pct / 100)

    if direction == 1:
        sell_strike = _round_strike(spot * 0.96)
        buy_strike = sell_strike - width
        sell_type, buy_type = "put", "put"
    else:
        sell_strike = _round_strike(spot * 1.04)
        buy_strike = sell_strike + width
        sell_type, buy_type = "call", "call"

    sell_price = _find_option_price(options, qdate, exp, sell_type, sell_strike, "bid")
    buy_price = _find_option_price(options, qdate, exp, buy_type, buy_strike, "ask")

    if pd.isna(buy_price) or pd.isna(sell_price):
        return {"pnl": np.nan, "return_pct": np.nan, "reason": "no_option_data"}

    credit = sell_price - buy_price
    if credit <= 0:
        return {"pnl": np.nan, "return_pct": np.nan, "reason": "negative_credit"}

    slippage = (buy_price + sell_price) * OPTION_SLIPPAGE_PCT / 100
    commission = 2 * COMMISSION_PER_CONTRACT
    net_credit = (credit - slippage) * CONTRACT_MULTIPLIER - commission

    expiry_close = trade["expiry_close"]
    max_loss = width * CONTRACT_MULTIPLIER

    if direction == 1:
        spread_val = max(0, sell_strike - expiry_close) - max(0, buy_strike - expiry_close)
    else:
        spread_val = max(0, expiry_close - sell_strike) - max(0, expiry_close - buy_strike)

    spread_loss = spread_val * CONTRACT_MULTIPLIER
    pnl = net_credit - spread_loss
    return_pct = pnl / net_credit * 100 if net_credit > 0 else 0

    return {
        "pnl": round(pnl, 2),
        "return_pct": round(return_pct, 2),
        "net_credit": round(net_credit, 2),
        "sell_strike": sell_strike,
        "buy_strike": buy_strike,
        "credit": round(credit, 4),
        "reason": "ok",
    }


def _price_iron_condor(trade: dict, options: pd.DataFrame, config: WeeklyConfig) -> dict:
    spot = trade["spot_price"]
    target = trade.get("target_strike", spot)
    exp = trade["expiration"]
    qdate = trade["quote_date"]

    width = _round_strike(spot * config.spread_width_pct / 100)

    short_put = _round_strike(target - width * 0.5)
    long_put = short_put - width
    short_call = _round_strike(target + width * 0.5)
    long_call = short_call + width

    lp = _find_option_price(options, qdate, exp, "put", long_put, "ask")
    sp = _find_option_price(options, qdate, exp, "put", short_put, "bid")
    lc = _find_option_price(options, qdate, exp, "call", long_call, "ask")
    sc = _find_option_price(options, qdate, exp, "call", short_call, "bid")

    if any(pd.isna(x) for x in [lp, sp, lc, sc]):
        return {"pnl": np.nan, "return_pct": np.nan, "reason": "no_option_data"}

    credit = (sp + sc - lp - lc)
    if credit <= 0:
        return {"pnl": np.nan, "return_pct": np.nan, "reason": "negative_credit"}

    slippage = (lp + sp + lc + sc) * OPTION_SLIPPAGE_PCT / 100
    commission = 4 * COMMISSION_PER_CONTRACT
    net_credit = (credit - slippage) * CONTRACT_MULTIPLIER - commission

    expiry_close = trade["expiry_close"]
    put_loss = (max(0, short_put - expiry_close) - max(0, long_put - expiry_close)) * CONTRACT_MULTIPLIER
    call_loss = (max(0, expiry_close - short_call) - max(0, expiry_close - long_call)) * CONTRACT_MULTIPLIER
    pnl = net_credit - put_loss - call_loss
    return_pct = pnl / net_credit * 100 if net_credit > 0 else 0

    return {
        "pnl": round(pnl, 2),
        "return_pct": round(return_pct, 2),
        "net_credit": round(net_credit, 2),
        "short_put": short_put, "long_put": long_put,
        "short_call": short_call, "long_call": long_call,
        "credit": round(credit, 4),
        "reason": "ok",
    }


def run_backtest(config: WeeklyConfig | None = None, verbose: bool = True) -> dict:
    if config is None:
        config = WeeklyConfig()

    pred_df = load_predictions()
    feat_df = load_features_v2()

    feat_cols = ["net_gex", "put_gex", "call_gex", "net_dealer_delta", "gex_flip_dist_pct",
                 "iv_rank", "iv_percentile_rank", "atm_iv_mean", "atr_pct",
                 "oi_concentration", "call_put_ratio", "risk_reversal_25d",
                 "max_volume_strike"]

    merged = pred_df.merge(
        feat_df[["ticker", "quote_date", "expiration"] + [c for c in feat_cols if c in feat_df.columns]],
        on=["ticker", "quote_date", "expiration"],
        how="left",
    )

    if config.tickers:
        merged = merged[merged["ticker"].isin(config.tickers)].copy()

    merged = merged.sort_values(["ticker", "expiration", "quote_date"]).reset_index(drop=True)

    entry_mask = (merged["days_to_expiry"] >= ENTRY_DTE_MIN) & (merged["days_to_expiry"] <= ENTRY_DTE_MAX)
    weekly = merged[entry_mask].copy()

    weekly = weekly.groupby(["ticker", "expiration"]).last().reset_index()
    weekly = weekly.sort_values(["ticker", "expiration"]).reset_index(drop=True)

    if config.trade_start:
        weekly = weekly[weekly["quote_date"] >= config.trade_start]
    if config.trade_end:
        weekly = weekly[weekly["quote_date"] <= config.trade_end]
    weekly = weekly.reset_index(drop=True)

    if verbose:
        print(f"Generated {len(weekly)} potential weekly trades")

    needed_tickers = weekly["ticker"].unique()
    opts_index = {}
    for ticker in needed_tickers:
        opts = load_options(ticker)
        if opts.empty:
            continue
        if verbose:
            print(f"  Indexing options for {ticker}...")
        idx = {}
        for (qd, exp), sub in opts.groupby(["quote_date", "expiration"]):
            idx[(qd, exp)] = sub
        opts_index[ticker] = idx

    trades = []
    for _, row in weekly.iterrows():
        best_pred = 0.0
        best_dir = 0
        for model_col in ["lgb_return_pred", "xgb_return_pred", "pt_return_pred"]:
            val = row.get(model_col, np.nan)
            if pd.notna(val) and abs(val) > abs(best_pred):
                best_pred = val
        for dir_col in ["lgb_direction", "xgb_direction", "pt_direction"]:
            val = row.get(dir_col, 0)
            if pd.notna(val) and val != 0:
                best_dir = int(val)

        direction = best_dir
        if direction == 0:
            if abs(best_pred) > config.direction_threshold:
                direction = 1 if best_pred > 0 else -1

        if direction == 0:
            continue

        pred_return = best_pred if best_pred != 0 else row.get("ensemble_return_pred", 0)
        if pd.isna(pred_return):
            pred_return = 0
        confidence = min(1.0, abs(pred_return) / 2.0)
        if confidence < config.min_confidence:
            confidence = config.min_confidence

        strategy = _select_strategy(row, config)
        if strategy == "flat":
            continue

        trade = {
            "ticker": row["ticker"],
            "expiration": row["expiration"],
            "quote_date": row["quote_date"],
            "spot_price": row["spot_price"],
            "expiry_close": row["expiry_close"],
            "direction": direction,
            "pred_return": pred_return,
            "confidence": confidence,
            "strategy": strategy,
            "target_strike": row.get("max_volume_strike", row["spot_price"]),
            "net_gex": row.get("net_gex", 0),
            "iv_rank": row.get("iv_rank", 50),
            "days_to_expiry": row["days_to_expiry"],
        }

        ticker_idx = opts_index.get(trade["ticker"], {})
        opts = ticker_idx.get((trade["quote_date"], trade["expiration"]), pd.DataFrame())
        if opts.empty:
            continue

        if strategy == "debit_spread":
            pricing = _price_debit_spread(trade, opts, config)
        elif strategy == "credit_spread":
            pricing = _price_credit_spread(trade, opts, config)
        elif strategy == "iron_condor":
            pricing = _price_iron_condor(trade, opts, config)
        else:
            continue

        if pd.isna(pricing.get("pnl")):
            continue

        trade.update(pricing)
        trade["position_pnl"] = trade["pnl"] * config.position_pct * confidence
        trades.append(trade)

    if not trades:
        print("No valid trades")
        return {}

    trades_df = pd.DataFrame(trades)
    if verbose:
        print(f"Priced {len(trades_df)} trades")
        for strat in trades_df["strategy"].unique():
            sub = trades_df[trades_df["strategy"] == strat]
            print(f"  {strat}: {len(sub)} trades, avg return={sub['return_pct'].mean():.2f}%, "
                  f"win rate={len(sub[sub['pnl'] > 0]) / len(sub) * 100:.1f}%")

    results = _evaluate(trades_df, config, verbose)

    out_path = DATA_DIR / "weekly_options_backtest.parquet"
    trades_df.to_parquet(out_path, index=False)
    if verbose:
        print(f"\nSaved to {out_path}")

    return {"trades": trades_df, "metrics": results}


def _evaluate(trades_df: pd.DataFrame, config: WeeklyConfig, verbose: bool) -> dict:
    splits = {
        "all": trades_df,
        "train": trades_df[trades_df["quote_date"] <= config.train_end],
        "val": trades_df[(trades_df["quote_date"] > config.train_end) & (trades_df["quote_date"] <= config.val_end)],
        "test": trades_df[trades_df["quote_date"] > config.val_end],
    }

    all_metrics = {}
    for name, sub in splits.items():
        if sub.empty:
            continue
        pnl = sub["position_pnl"]
        winning = pnl[pnl > 0]
        losing = pnl[pnl < 0]

        win_rate = len(winning) / len(pnl) * 100 if len(pnl) > 0 else 0
        profit_factor = winning.sum() / abs(losing.sum()) if len(losing) > 0 and losing.sum() != 0 else np.inf
        total_pnl = pnl.sum()
        avg_pnl = pnl.mean()
        avg_return = sub["return_pct"].mean()

        cumulative = pnl.cumsum()
        running_max = cumulative.cummax()
        drawdown = cumulative - running_max
        max_dd = drawdown.min()

        ann_factor = 52
        sharpe = pnl.mean() / pnl.std() * np.sqrt(ann_factor) if pnl.std() > 0 else 0

        metrics = {
            "n_trades": len(sub),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(avg_pnl, 2),
            "avg_return_pct": round(avg_return, 2),
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 4) if profit_factor != np.inf else "inf",
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_dd, 2),
        }
        all_metrics[name] = metrics

        if verbose:
            print(f"\n  {name}:")
            for k, v in metrics.items():
                print(f"    {k:20s}: {v}")

    if verbose:
        print(f"\n  Per strategy breakdown:")
        for strat in trades_df["strategy"].unique():
            sub = trades_df[trades_df["strategy"] == strat]
            print(f"    {strat}: n={len(sub)}, avg_ret={sub['return_pct'].mean():.2f}%, "
                  f"win={len(sub[sub['pnl'] > 0]) / len(sub) * 100:.1f}%, "
                  f"total_pnl={sub['position_pnl'].sum():.2f}")

        print(f"\n  Per ticker breakdown:")
        for ticker in sorted(trades_df["ticker"].unique()):
            sub = trades_df[trades_df["ticker"] == ticker]
            print(f"    {ticker:6s}: n={len(sub)}, avg_ret={sub['return_pct'].mean():.2f}%, "
                  f"win={len(sub[sub['pnl'] > 0]) / len(sub) * 100:.1f}%")

    return all_metrics


def run_all_configs(verbose: bool = True) -> dict:
    configs = {
        "auto_all": WeeklyConfig(
            strategy_type="auto",
            min_confidence=0.3,
            spread_width_pct=1.5,
        ),
        "auto_etf": WeeklyConfig(
            strategy_type="auto",
            tickers=list(ETF_TICKERS),
            min_confidence=0.3,
            spread_width_pct=1.5,
        ),
        "debit_spread_all": WeeklyConfig(
            strategy_type="debit_spread",
            min_confidence=0.3,
            spread_width_pct=1.5,
        ),
        "credit_spread_all": WeeklyConfig(
            strategy_type="credit_spread",
            min_confidence=0.3,
            spread_width_pct=1.5,
        ),
        "iron_condor_all": WeeklyConfig(
            strategy_type="iron_condor",
            min_confidence=0.3,
            spread_width_pct=1.5,
        ),
        "auto_etf_tight": WeeklyConfig(
            strategy_type="auto",
            tickers=list(ETF_TICKERS),
            min_confidence=0.5,
            spread_width_pct=1.0,
        ),
    }

    all_results = {}
    summary = []

    for name, cfg in configs.items():
        print(f"\n{'=' * 60}")
        print(f"  Config: {name}")
        print(f"{'=' * 60}")
        print(f"  Config: {name}")
        print(f"{'=' * 60}")
        result = run_backtest(cfg, verbose=verbose)
        all_results[name] = result
        if "metrics" in result:
            for split_name, m in result["metrics"].items():
                row = {"config": name, "split": split_name, **m}
                summary.append(row)

    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.to_parquet(DATA_DIR / "weekly_strategy_summary.parquet", index=False)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  SUMMARY: Test Set Comparison")
            print(f"{'=' * 60}")
            test_summary = summary_df[summary_df["split"] == "test"]
            if not test_summary.empty:
                for col in ["n_trades", "avg_return_pct", "win_rate_pct", "sharpe", "profit_factor", "max_drawdown"]:
                    if col in test_summary.columns:
                        print(f"\n  {col}:")
                        for _, row in test_summary.iterrows():
                            print(f"    {row['config']:25s}: {row[col]}")

    return all_results


if __name__ == "__main__":
    run_all_configs()
