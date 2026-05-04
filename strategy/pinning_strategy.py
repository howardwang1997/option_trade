from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

ETF_TICKERS = {"SPY", "QQQ", "GLD"}
TARGET_CANDIDATE = "max_volume_strike"


@dataclass
class BacktestConfig:
    strategy_type: Literal["stock", "vertical_spread", "butterfly", "iron_condor"] = "stock"
    target_candidate: str = TARGET_CANDIDATE
    etf_only: bool = True
    min_oi_concentration: float = 0.05
    max_atr_pct: float = 1.5
    min_dte: int = 3
    max_dte: int = 5
    stock_slippage_bps: float = 5.0
    option_slippage_pct: float = 0.5
    stock_commission_bps: float = 1.0
    option_commission_per_contract: float = 0.65
    contract_multiplier: int = 100
    position_size_method: Literal["equal", "confidence_scaled", "kelly"] = "equal"
    base_position_pct: float = 1.0
    spread_width_pct: float = 2.0
    allowed_tickers: list[str] | None = None
    train_end: str = "2019-12-31"
    val_end: str = "2022-12-31"


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    return df


_options_cache: dict[str, pd.DataFrame] = {}


def load_options(ticker: str, dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
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

    if dates is not None:
        df = df[df["quote_date"].isin(dates)]

    _options_cache[ticker] = df
    return df


def generate_weekly_trades(df: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    if config.allowed_tickers:
        df = df[df["ticker"].isin(config.allowed_tickers)].copy()
    elif config.etf_only:
        df = df[df["ticker"].isin(ETF_TICKERS)].copy()

    df = df.sort_values(["ticker", "expiration", "quote_date"]).reset_index(drop=True)

    trades = []
    for (ticker, expiration), group in df.groupby(["ticker", "expiration"]):
        entry_rows = group[
            (group["days_to_expiry"] >= config.min_dte)
            & (group["days_to_expiry"] <= config.max_dte)
        ]
        if entry_rows.empty:
            continue

        entry = entry_rows.iloc[-1]
        candidate_val = entry[config.target_candidate]
        spot = entry["spot_price"]
        dte = entry["days_to_expiry"]
        atr_pct = entry["atr_pct"]
        oi_conc = entry["oi_concentration"]

        if pd.isna(candidate_val) or candidate_val <= 0:
            continue
        if atr_pct > config.max_atr_pct:
            continue
        if oi_conc < config.min_oi_concentration:
            continue

        target_dist_pct = (candidate_val - spot) / spot * 100
        if abs(target_dist_pct) < 0.3:
            continue

        direction = 1 if target_dist_pct > 0 else -1

        vol_score = max(0.0, min(1.0, (config.max_atr_pct - atr_pct) / config.max_atr_pct))
        conc_score = min(1.0, oi_conc / 0.20)
        target_score = min(1.0, abs(target_dist_pct) / 5.0)
        direction_penalty = 1.0 if direction == 1 else 0.7
        confidence = (0.40 * vol_score + 0.35 * conc_score + 0.25 * target_score) * direction_penalty
        confidence = round(max(0.0, min(1.0, confidence)), 4)

        entry_dte_row = group[group["days_to_expiry"] == dte]
        entry_price = entry_dte_row.iloc[0]["spot_price"] if not entry_dte_row.empty else spot

        expiry_row = group[group["days_to_expiry"] == group["days_to_expiry"].min()]
        expiry_close = expiry_row.iloc[0]["expiry_close"] if not expiry_row.empty else np.nan
        expiry_high = expiry_row.iloc[0]["expiry_high"] if not expiry_row.empty else np.nan
        expiry_low = expiry_row.iloc[0]["expiry_low"] if not expiry_row.empty else np.nan

        trades.append({
            "ticker": ticker,
            "expiration": expiration,
            "entry_date": entry["quote_date"],
            "entry_price": entry_price,
            "entry_dte": int(dte),
            "direction": direction,
            "confidence": confidence,
            "target_strike": candidate_val,
            "atr_pct": atr_pct,
            "oi_concentration": oi_conc,
            "expiry_close": expiry_close,
            "expiry_high": expiry_high,
            "expiry_low": expiry_low,
        })

    return pd.DataFrame(trades).sort_values(["ticker", "expiration"]).reset_index(drop=True) if trades else pd.DataFrame()


def price_stock_pnl(trade: pd.Series, config: BacktestConfig) -> dict:
    entry = trade["entry_price"]
    exit_price = trade["expiry_close"]
    direction = trade["direction"]

    slippage = entry * config.stock_slippage_bps / 10000
    commission = entry * config.stock_commission_bps / 10000

    effective_entry = entry + slippage + commission
    effective_exit = exit_price - slippage - commission

    if direction == 1:
        raw_return = (exit_price - entry) / entry * 100
        net_return = (effective_exit - effective_entry) / effective_entry * 100
    else:
        raw_return = (entry - exit_price) / entry * 100
        net_return = (effective_entry - effective_exit) / effective_entry * 100

    return {
        "raw_return_pct": round(raw_return, 4),
        "net_return_pct": round(net_return, 4),
        "total_cost_pct": round((slippage + commission) * 2 / entry * 100, 4),
    }


def _find_option_price(options: pd.DataFrame, quote_date, expiration, opt_type: str,
                       strike: float, side: str = "mid") -> float:
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
        ]["strike"] if not options.empty else pd.Series(dtype=float)
        if strikes.empty:
            return np.nan
        nearest_idx = (strikes - strike).abs().idxmin()
        match = options.loc[[nearest_idx]]
        if match.iloc[0]["type"] != opt_type:
            return np.nan

    row = match.iloc[0]
    bid = float(row["bid"]) if pd.notna(row["bid"]) else 0.0
    ask = float(row["ask"]) if pd.notna(row["ask"]) else 0.0

    if side == "bid":
        return bid
    elif side == "ask":
        return ask
    else:
        return (bid + ask) / 2 if bid + ask > 0 else np.nan


def clear_options_cache():
    _options_cache.clear()


def _round_strike(price: float, tick_size: float = 0.5) -> float:
    return round(price / tick_size) * tick_size


def price_vertical_spread(trade: pd.Series, options: pd.DataFrame, config: BacktestConfig) -> dict:
    spot = trade["entry_price"]
    direction = trade["direction"]
    target = trade["target_strike"]
    exp = trade["expiration"]
    qdate = trade["entry_date"]

    spread_width = spot * config.spread_width_pct / 100
    spread_width = _round_strike(spread_width)

    if direction == 1:  # bullish: buy lower strike call, sell higher strike call (debit spread)
        buy_strike = _round_strike(spot * 0.98)
        sell_strike = buy_strike + spread_width
        buy_type, sell_type = "call", "call"
        buy_side, sell_side = "ask", "bid"
    else:  # bearish: buy higher strike put, sell lower strike put (debit put spread)
        sell_strike = _round_strike(spot * 0.98)
        buy_strike = sell_strike + spread_width
        buy_type, sell_type = "put", "put"
        buy_side, sell_side = "ask", "bid"

    buy_price = _find_option_price(options, qdate, exp, buy_type, buy_strike, buy_side)
    sell_price = _find_option_price(options, qdate, exp, sell_type, sell_strike, sell_side)

    if pd.isna(buy_price) or pd.isna(sell_price):
        return {"net_return_pct": np.nan, "max_profit_pct": np.nan, "reason": "no_option_data"}

    debit = (buy_price - sell_price) * config.contract_multiplier
    if debit <= 0:
        return {"net_return_pct": np.nan, "max_profit_pct": np.nan, "reason": "negative_debit"}

    slippage_cost = (buy_price + sell_price) * config.option_slippage_pct / 100 * config.contract_multiplier
    commission = 2 * config.option_commission_per_contract * config.contract_multiplier
    total_cost = debit + slippage_cost + commission

    expiry_close = trade["expiry_close"]

    if direction == 1:
        buy_intrinsic = max(0, expiry_close - buy_strike)
        sell_intrinsic = max(0, expiry_close - sell_strike)
    else:
        buy_intrinsic = max(0, buy_strike - expiry_close)
        sell_intrinsic = max(0, sell_strike - expiry_close)

    exit_value = (buy_intrinsic - sell_intrinsic) * config.contract_multiplier
    max_profit = (spread_width * config.contract_multiplier) - total_cost
    pnl = exit_value - total_cost

    net_return = pnl / total_cost * 100 if total_cost > 0 else 0.0
    max_profit_pct = max_profit / total_cost * 100 if total_cost > 0 else 0.0

    return {
        "net_return_pct": round(net_return, 4),
        "max_profit_pct": round(max_profit_pct, 4),
        "debit": round(debit, 2),
        "total_cost": round(total_cost, 2),
        "pnl": round(pnl, 2),
        "long_strike": buy_strike,
        "short_strike": sell_strike,
        "long_type": buy_type,
        "long_entry_price": round(buy_price, 4),
        "short_entry_price": round(sell_price, 4),
        "reason": "ok",
    }


def price_butterfly(trade: pd.Series, options: pd.DataFrame, config: BacktestConfig) -> dict:
    spot = trade["entry_price"]
    direction = trade["direction"]
    target = trade["target_strike"]
    exp = trade["expiration"]
    qdate = trade["entry_date"]

    center = _round_strike(target)
    wing_width = _round_strike(spot * config.spread_width_pct / 100)
    lower = center - wing_width
    upper = center + wing_width

    opt_type = "call" if direction == 1 else "put"

    wing_low = _find_option_price(options, qdate, exp, opt_type, lower, "ask")
    mid_price = _find_option_price(options, qdate, exp, opt_type, center, "bid")
    wing_high = _find_option_price(options, qdate, exp, opt_type, upper, "ask")

    if pd.isna(wing_low) or pd.isna(mid_price) or pd.isna(wing_high):
        return {"net_return_pct": np.nan, "reason": "no_option_data"}

    debit = (wing_low + wing_high - 2 * mid_price) * config.contract_multiplier
    if debit <= 0:
        return {"net_return_pct": np.nan, "reason": "negative_debit"}

    slippage_cost = (wing_low + wing_high + 2 * mid_price) * config.option_slippage_pct / 100 * config.contract_multiplier
    commission = 4 * config.option_commission_per_contract * config.contract_multiplier
    total_cost = debit + slippage_cost + commission

    expiry_close = trade["expiry_close"]

    if direction == 1:
        low_val = max(0, expiry_close - lower)
        mid_val = max(0, expiry_close - center)
        high_val = max(0, expiry_close - upper)
    else:
        low_val = max(0, lower - expiry_close)
        mid_val = max(0, center - expiry_close)
        high_val = max(0, upper - expiry_close)

    exit_value = (low_val + high_val - 2 * mid_val) * config.contract_multiplier
    max_profit = (wing_width * config.contract_multiplier) - total_cost
    pnl = exit_value - total_cost

    net_return = pnl / total_cost * 100 if total_cost > 0 else 0.0
    max_profit_pct = max_profit / total_cost * 100 if total_cost > 0 else 0.0

    return {
        "net_return_pct": round(net_return, 4),
        "max_profit_pct": round(max_profit_pct, 4),
        "debit": round(debit, 2),
        "total_cost": round(total_cost, 2),
        "pnl": round(pnl, 2),
        "center_strike": center,
        "lower_strike": lower,
        "upper_strike": upper,
        "reason": "ok",
    }


def price_iron_condor(trade: pd.Series, options: pd.DataFrame, config: BacktestConfig) -> dict:
    spot = trade["entry_price"]
    target = trade["target_strike"]
    exp = trade["expiration"]
    qdate = trade["entry_date"]

    wing_width = _round_strike(spot * config.spread_width_pct / 100)
    short_put = _round_strike(target - wing_width)
    long_put = short_put - wing_width
    short_call = _round_strike(target + wing_width)
    long_call = short_call + wing_width

    long_put_price = _find_option_price(options, qdate, exp, "put", long_put, "ask")
    short_put_price = _find_option_price(options, qdate, exp, "put", short_put, "bid")
    long_call_price = _find_option_price(options, qdate, exp, "call", long_call, "ask")
    short_call_price = _find_option_price(options, qdate, exp, "call", short_call, "bid")

    if any(pd.isna(x) for x in [long_put_price, short_put_price, long_call_price, short_call_price]):
        return {"net_return_pct": np.nan, "reason": "no_option_data"}

    credit = (short_put_price + short_call_price - long_put_price - long_call_price) * config.contract_multiplier
    if credit <= 0:
        return {"net_return_pct": np.nan, "reason": "negative_credit"}

    slippage_cost = (long_put_price + short_put_price + long_call_price + short_call_price) * config.option_slippage_pct / 100 * config.contract_multiplier
    commission = 4 * config.option_commission_per_contract * config.contract_multiplier
    net_credit = credit - slippage_cost - commission

    expiry_close = trade["expiry_close"]

    put_spread_val = max(0, short_put - expiry_close) - max(0, long_put - expiry_close)
    call_spread_val = max(0, expiry_close - short_call) - max(0, expiry_close - long_call)
    spread_loss = (put_spread_val + call_spread_val) * config.contract_multiplier

    max_loss = (wing_width * config.contract_multiplier) - net_credit
    pnl = net_credit - spread_loss
    net_return = pnl / net_credit * 100 if net_credit > 0 else 0.0
    max_profit_pct = net_credit / max_loss * 100 if max_loss > 0 else 0.0

    return {
        "net_return_pct": round(net_return, 4),
        "max_profit_pct": round(max_profit_pct, 4),
        "credit": round(credit, 2),
        "net_credit": round(net_credit, 2),
        "pnl": round(pnl, 2),
        "short_put": short_put,
        "long_put": long_put,
        "short_call": short_call,
        "long_call": long_call,
        "reason": "ok",
    }


def compute_position_size(confidence: float, method: str, base_pct: float) -> float:
    if method == "equal":
        return base_pct
    elif method == "confidence_scaled":
        return base_pct * confidence
    elif method == "kelly":
        kelly = confidence * 2 - 1
        kelly = max(0.1, min(1.0, kelly))
        return base_pct * kelly * 0.5
    return base_pct


def compute_metrics(returns: pd.Series, name: str = "") -> dict:
    if len(returns) == 0:
        return {}
    cumulative = (1 + returns / 100).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max * 100

    winning = returns[returns > 0]
    losing = returns[returns < 0]
    win_rate = len(winning) / len(returns) * 100 if len(returns) > 0 else 0
    profit_factor = winning.sum() / abs(losing.sum()) if len(losing) > 0 and losing.sum() != 0 else np.inf
    avg_win = winning.mean() if len(winning) > 0 else 0
    avg_loss = losing.mean() if len(losing) > 0 else 0
    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    total_return = (cumulative.iloc[-1] - 1) * 100 if len(cumulative) > 0 else 0
    years = len(returns) / 52 if len(returns) > 0 else 1
    cagr = ((cumulative.iloc[-1]) ** (1 / years) - 1) * 100 if len(cumulative) > 0 and years > 0 else 0
    ann_vol = returns.std() * np.sqrt(52) if len(returns) > 1 else 0
    sharpe = (returns.mean() / returns.std() * np.sqrt(52)) if returns.std() > 0 else 0
    max_dd = drawdown.min() if len(drawdown) > 0 else 0

    metrics = {
        "name": name,
        "n_trades": len(returns),
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 4) if profit_factor != np.inf else "inf",
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_return_pct": round(returns.mean(), 4),
        "avg_win_pct": round(avg_win, 4),
        "avg_loss_pct": round(avg_loss, 4),
        "payoff_ratio": round(payoff_ratio, 4) if payoff_ratio != np.inf else "inf",
        "ann_volatility_pct": round(ann_vol, 2),
    }

    return metrics


def run_backtest(config: BacktestConfig | None = None, verbose: bool = True) -> dict:
    if config is None:
        config = BacktestConfig()

    df = load_features()
    trades = generate_weekly_trades(df, config)

    if trades.empty:
        print("No trades generated")
        return {}

    if verbose:
        print(f"Generated {len(trades)} trades")

    options_cache: dict[str, pd.DataFrame] = {}

    if config.strategy_type != "stock":
        needed_tickers = trades["ticker"].unique()
        needed_dates = trades["entry_date"].unique()
        for t in needed_tickers:
            print(f"  Loading options for {t}...")
            load_options(t, pd.DatetimeIndex(needed_dates))

    results = []
    for idx, trade in trades.iterrows():
        if config.strategy_type == "stock":
            pricing = price_stock_pnl(trade, config)
        else:
            opts = load_options(trade["ticker"])
            if opts.empty:
                pricing = {"net_return_pct": np.nan, "reason": "no_options_data"}
            elif config.strategy_type == "vertical_spread":
                pricing = price_vertical_spread(trade, opts, config)
            elif config.strategy_type == "butterfly":
                pricing = price_butterfly(trade, opts, config)
            elif config.strategy_type == "iron_condor":
                pricing = price_iron_condor(trade, opts, config)
            else:
                pricing = {"net_return_pct": np.nan, "reason": "unknown_strategy"}

        row = {**trade.to_dict(), **pricing}
        results.append(row)
        if verbose and (idx + 1) % 200 == 0:
            print(f"  Priced {idx + 1}/{len(trades)} trades")

    results_df = pd.DataFrame(results)
    valid = results_df[results_df["net_return_pct"].notna()].copy()

    if valid.empty:
        print("No valid trades after pricing")
        return {"trades": results_df}

    position_sizes = valid["confidence"].apply(
        lambda c: compute_position_size(c, config.position_size_method, config.base_position_pct)
    )
    valid["position_return_pct"] = valid["net_return_pct"] * position_sizes

    train_mask = valid["entry_date"] <= config.train_end
    val_mask = (valid["entry_date"] > config.train_end) & (valid["entry_date"] <= config.val_end)
    test_mask = valid["entry_date"] > config.val_end

    splits = {
        "all": valid,
        "train (2010-2019)": valid[train_mask],
        "val (2020-2022)": valid[val_mask],
        "test (2023+)": valid[test_mask],
    }

    all_metrics = {}
    for name, split_df in splits.items():
        if split_df.empty:
            continue
        m = compute_metrics(split_df["position_return_pct"], name)
        all_metrics[name] = m
        if verbose:
            print(f"\n{'='*50}")
            print(f"  {name} | {config.strategy_type}")
            print(f"{'='*50}")
            for k, v in m.items():
                print(f"  {k:25s}: {v}")

    out_path = DATA_DIR / f"backtest_{config.strategy_type}.parquet"
    results_df.to_parquet(out_path, index=False)
    if verbose:
        print(f"\nSaved trades to {out_path}")

    return {"metrics": all_metrics, "trades": results_df}


def run_all_strategies(verbose: bool = True) -> dict:
    configs = {
        "stock_etf_conservative": BacktestConfig(
            strategy_type="stock",
            etf_only=True,
            min_oi_concentration=0.05,
            max_atr_pct=1.0,
            min_dte=3,
            max_dte=5,
            stock_slippage_bps=5.0,
            option_slippage_pct=0.5,
            position_size_method="confidence_scaled",
        ),
        "stock_etf_moderate": BacktestConfig(
            strategy_type="stock",
            etf_only=True,
            min_oi_concentration=0.03,
            max_atr_pct=1.5,
            min_dte=3,
            max_dte=5,
            stock_slippage_bps=5.0,
            position_size_method="confidence_scaled",
        ),
        "vertical_spread_etf": BacktestConfig(
            strategy_type="vertical_spread",
            etf_only=True,
            min_oi_concentration=0.05,
            max_atr_pct=1.0,
            min_dte=3,
            max_dte=5,
            option_slippage_pct=0.5,
            spread_width_pct=2.0,
            position_size_method="equal",
        ),
        "butterfly_etf": BacktestConfig(
            strategy_type="butterfly",
            etf_only=True,
            min_oi_concentration=0.05,
            max_atr_pct=1.0,
            min_dte=3,
            max_dte=5,
            option_slippage_pct=0.5,
            spread_width_pct=2.0,
            position_size_method="equal",
        ),
        "iron_condor_etf": BacktestConfig(
            strategy_type="iron_condor",
            etf_only=True,
            min_oi_concentration=0.05,
            max_atr_pct=1.0,
            min_dte=3,
            max_dte=5,
            option_slippage_pct=0.5,
            spread_width_pct=2.0,
            position_size_method="equal",
        ),
    }

    all_results = {}
    summary_rows = []
    for name, cfg in configs.items():
        if verbose:
            print(f"\n{'#'*70}")
            print(f"# Strategy: {name}")
            print(f"{'#'*70}")
        result = run_backtest(cfg, verbose=verbose)
        all_results[name] = result

        if "metrics" in result:
            for split_name, m in result["metrics"].items():
                row = {"strategy": name, **m}
                summary_rows.append(row)

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        if verbose:
            print(f"\n{'#'*70}")
            print(f"# SUMMARY COMPARISON")
            print(f"{'#'*70}")
            for col in ["n_trades", "total_return_pct", "cagr_pct", "win_rate_pct",
                        "sharpe_ratio", "max_drawdown_pct", "profit_factor"]:
                if col in summary.columns:
                    pivot = summary.pivot(index="strategy", columns="name", values=col)
                    print(f"\n--- {col} ---")
                    print(pivot.round(2).to_string())

        summary.to_parquet(DATA_DIR / "backtest_summary.parquet", index=False)

    return all_results


if __name__ == "__main__":
    run_all_strategies()
