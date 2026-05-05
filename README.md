# option_trade

US equity & ETF options and OHLCV data pipeline.

## Data Coverage

**Tickers:** NVDA, GOOG, GOOGL, MSFT, AMZN, AAPL, SPY, QQQ, GLD

| Dataset | Source | Range |
|---------|--------|-------|
| OHLCV | Repo snapshots + yfinance | 1999 — present |
| Options chain | Repo snapshots | 2010 — present |

## Setup

```bash
conda create -n snapshot-pipeline python=3.11 -y
conda activate snapshot-pipeline
pip install -r requirements.txt
```

Configure `.env` with R2 credentials and data path:

```
DATA_DIR=/path/to/daily-snapshots
R2_ENDPOINT=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET=wall-street-baos
R2_REGION=apac
```

## Usage

```bash
# Run full pipeline (extract + yfinance + write parquet + upload to R2)
PHASE=all python pipeline.py

# Run individual phases
PHASE=ohlcv    python pipeline.py   # OHLCV extraction only
PHASE=options  python pipeline.py   # Options extraction only
PHASE=upload   python pipeline.py   # Upload existing output/ to R2
```

## Output Structure

```
output/
├── manifest.json
├── AAPL/
│   ├── ohlcv.parquet       # date, open, high, low, close, volume, source
│   └── options.parquet     # quote_date, expiration, type, strike, bid, ask, volume, open_interest, delta, gamma, theta, vega, implied_volatility
├── NVDA/ ...
└── ...
```

## R2 Storage

Parquet files are uploaded to Cloudflare R2 under `data/{TICKER}/`.

## Research: Option Pinning Effect

### Theory

On option expiration days, stock prices tend to gravitate toward strikes where
option activity is concentrated, causing those options to expire worthless.
The "target" strike is **not** classic max-pain — it is dynamically determined
by OI distribution, volume, and market structure.

### Phase 1 Findings (Completed)

#### Best Pinning Indicator

| Rank | Candidate | Mean Dist | <1% Hit | <2% Hit | <3% Hit |
|------|-----------|-----------|---------|---------|---------|
| 1 | **max_volume_strike** | 4.66% | 38.5% | 76.3% | 88.0% |
| 2 | oi_mass_center | 6.40% | 17.9% | — | — |
| 3 | max_call_oi_strike | 6.86% | 24.0% | — | — |
| 8 | max_pain_strike | 34.5% | 0.2% | — | **useless** |

**Conclusion:** `max_volume_strike` (highest volume strike) is the best
pinning indicator. Max-pain does not work.

#### Best Tickers (DTE=1, max_volume_strike, unfiltered)

| Ticker | <1% Hit | <2% Hit | <3% Hit | <5% Hit |
|--------|---------|---------|---------|---------|
| **SPY** | 66.7% | 89.0% | 95.3% | 98.6% |
| **QQQ** | 56.3% | 82.3% | 92.9% | 97.8% |
| **GLD** | 61.0% | 84.4% | 93.4% | 97.3% |
| MSFT | 48.8% | 74.3% | 88.3% | 95.9% |
| AAPL | 47.4% | 75.3% | 86.7% | 95.9% |
| NVDA | 28.5% | 51.6% | 68.5% | 85.3% |

ETFs (SPY/QQQ/GLD) consistently outperform individual stocks.

#### High-Probability Factor Combinations

Simple rules with **>95% hit rate (<2%)**:

| Ticker | DTE | Condition | Hit <2% | N | Mean Dist |
|--------|-----|-----------|---------|---|-----------|
| SPY | 1 | atr≤1% + oi_conc≥5% | **95.8%** | 577 | 0.72% |
| QQQ | 1 | atr≤1% + oi_conc≥10% | **95.4%** | 108 | 0.67% |
| SPY | 3 | atr≤1% + oi_conc≥3% | **89.4%** | 540 | 1.04% |
| SPY | 4 | atr≤1% + oi_conc≥5% | **86.4%** | 463 | 1.13% |
| SPY | 5 | atr≤1% + oi_conc≥5% | **84.6%** | 409 | 1.16% |

#### Directional Edge (DTE 3-5, target above spot → go long)

| DTE | Ticker | Target above → price up | Mean move |
|-----|--------|------------------------|-----------|
| 4 | QQQ | **61.7%** | +1.10% |
| 5 | QQQ | **63.7%** | +1.74% |
| 4 | SPY | **60.3%** | +0.77% |
| 5 | SPY | **60.2%** | +0.84% |

#### RF Feature Importance (predicting pinning <1%)

Top factors: `atr_pct` > `realized_vol_20d` > `days_to_expiry` > `atm_iv` >
`dist_to_resistance_pct` > `price_vs_sma50_pct`

**Core insight:** Low volatility environment is the #1 predictor of pinning success.

#### Time Stability

SPY/QQQ/GLD show consistent pinning effect across 2010-2026.
NVDA weakens during high-vol years (2022, 2026) but recovers.

#### Price Regime

SPY at lower price quintiles shows stronger pinning (76% vs 66% hit rate).
NVDA shows no significant price-regime dependency (consistently ~28%).

### Phase 2 (Completed)

#### Step 5: `strategy/signals.py` — Signal Generation

Dynamic pinning signal with conditions from Phase 1:
- Long signal: ETF (SPY/QQQ/GLD) + atr≤1.5% + oi_conc≥5% + max_volume_strike above spot
- Short signal: mirror condition (weaker, ~42% accuracy)
- Confidence scoring: weighted by volatility (40%), OI concentration (35%), target distance (25%)
- Direction penalty: short signals get 0.7x confidence scaling
- Three configs: conservative (atr≤1%, oi_conc≥5%), moderate (atr≤1.5%, oi_conc≥3%), aggressive (all tickers)

#### Step 6: `strategy/pinning_strategy.py` — Strategy Backtest

Full backtest engine with stock and options strategies:

**Stock Strategy (Baseline)**
- Enter DTE 3-5, exit at expiry (hold to Friday close)
- Direction based on max_volume_strike vs spot
- Slippage: 5 bps entry + 5 bps exit, commission: 1 bps each side
- Position sizing: confidence-scaled

**Options Strategies**
- Vertical spread (debit call/put spread), 2% width
- Butterfly spread (3-strike centered on target)
- Iron condor (sell straddle at target, buy wings)
- Option pricing: actual bid/ask from options.parquet
- Slippage: 0.5% of premium, commission: $0.65/contract

**Metrics**: Sharpe, win rate, profit factor, max drawdown, CAGR, payoff ratio

#### Stock Strategy Results (ETF Only, Conservative)

| Period | Trades | Total Ret | CAGR | Win Rate | Sharpe | Max DD | Profit Factor |
|--------|--------|-----------|------|----------|--------|--------|---------------|
| All | 652 | +10.3% | 0.8% | 50.5% | 0.31 | -7.0% | 1.13 |
| Train 2010-2019 | 386 | -3.7% | -0.5% | 48.7% | -0.19 | -7.0% | 0.92 |
| Val 2020-2022 | 72 | 0.0% | 0.0% | 48.6% | 0.02 | -4.1% | 1.01 |
| **Test 2023+** | **194** | **+14.6%** | **3.7%** | **54.6%** | **1.39** | **-2.7%** | **1.71** |

Key finding: strategy alpha concentrated in 2023+ period. Train/val periods flat to negative.

#### Options Strategy Notes

- Vertical spreads on DTE 3-5 are challenging due to wide spread widths vs short holding period
- Butterfly spreads centered on target strike have low win rate (~16% for SPY)
- Iron condors (sell premium at target) are structurally better for pinning thesis
- Options strategies need RL optimization to improve entry/exit timing

### Phase 3 (Completed)

#### Step 7: `rl/env.py` — RL Environment

- Gymnasium env, episode = one expiration cycle per (ticker, expiration)
- State (39 dim): 28 raw features + 9 ticker one-hot + current position + unrealized PnL
- Feature normalization: strikes → % distance from spot, OI/volume → log scale, DTE → /10
- Action: 3 discrete (short=0, flat=1, long=2)
- Reward: realized PnL (slippage 10bps round-trip) × reward_scaling + small unrealized incentive
- Penalty for no-trade episodes
- Episodes: train=1766, val=977, test=2029

#### Step 8: `rl/train.py` — PPO Training & Evaluation

- PPO (stable-baselines3), MLP [128, 128, 64], device=cpu
- MetricsCallback: evaluate every 4096 steps on val set, save best model
- 5-seed ensemble training (200k steps each)
- Rule-based baseline comparison (max_volume_strike direction, ±0.5% threshold)

#### RL Ensemble Results (5 seeds, test set)

| Seed | Mean PnL | Win Rate | Sharpe | Trades/ep |
|------|----------|----------|--------|-----------|
| 0 | -0.136% | 45.2% | -0.540 | 1.19 |
| 1 | -0.014% | 47.6% | -0.050 | 1.21 |
| 2 | -0.049% | 47.8% | -0.165 | 1.26 |
| 3 | -0.088% | 46.0% | -0.323 | 1.13 |
| 4 | -0.056% | 47.0% | -0.212 | 1.16 |
| **Ensemble avg** | **-0.068%** | **46.7%** | **-0.258** | **1.19** |
| Rule baseline | -0.057% | 34.2% | -0.246 | — |

**Conclusion**: PPO does not significantly outperform the simple rule-based baseline.
The short episode length (3-5 steps) and sparse reward signal limit RL effectiveness.
Win rate improves (46.7% vs 34.2%) but PnL is slightly worse due to more frequent smaller trades.

### Possible Next Steps

- Reward shaping: risk-adjusted reward (Sharpe per episode), position-dependent reward
- Confidence-gated action: combine RL with Phase 2 signal confidence, only trade when both agree
- Alternative algorithms: SAC, DQN instead of PPO
- Longer episode horizon: include pre-expiration week for more context
- Use RL to optimize strike selection for options strategies (iron condor, butterfly)
- Continuous action space for position sizing

### Phase 4 (Completed)

#### Step 9: `features/advanced_features.py` — Advanced Feature Engineering

Extracted 56 new features from 102.6M raw options contract rows across 9 tickers:

| Category | Features | Description |
|----------|----------|-------------|
| GEX + Dealer | 10 | Net gamma exposure, GEX flip point, dealer delta, vega/theta exposure |
| IV Surface | 12 | IV at fixed deltas, risk reversal, skew, smile, term structure |
| IV Rank | 2 | Percentile rank and IV rank vs 252-day history |
| OI Dynamics | 7 | Daily/3-day OI change, concentration change, CPR change |
| Technicals | 13 | MACD, Bollinger Bands, OBV, multi-period returns |
| Cross-asset | 2 | 60d beta and correlation to SPY |

Output: `data/features_v2.parquet` — 34,213 rows × 98 columns

#### Step 10: `models/predict.py` — Multi-Model Prediction

Three regression models predicting weekly return magnitude using all 76 features:

| Model | CV MAE | Test Direction Acc | Top Features |
|-------|--------|--------------------|--------------|
| LightGBM | 1.20 | 42.4% | price_vs_sma5_pct, vol_ratio_5_20, oi_concentration_change, put_gex |
| XGBoost | 1.26 | 45.1% | price_vs_sma5_pct, obv_sma20, total_oi, put_gex, iv_put_25d |
| PyTorch MLP | 1.16 | — | return_10d, return_5d, atr_14, realized_vol_20d |

New GEX/IV features ranked in top 10: put_gex, iv_near_term, atm_gamma, iv_term_structure_slope

#### Step 11: `strategy/weekly_strategy.py` — Weekly Options Strategy

Weekly options: enter Tue/Wed (DTE 3-4), hold to Friday expiry, auto-select strategy type.

**Test Set Results (2023+)**

| Config | Trades | Avg Return | Win Rate | Sharpe |
|--------|--------|-----------|----------|--------|
| debit_spread_all | 1850 | -1.84% | 51.7% | -0.49 |
| credit_spread_all | 1410 | -111.6% | 82.8% | -0.65 |
| iron_condor_all | 1836 | -29.4% | 45.3% | -1.12 |
| auto_etf | 276 | -18.3% | 51.1% | -1.22 |

**Conclusion**: All options strategies lose money. The core problem is that short-DTE options
have large bid-ask spreads relative to expected price moves (1-2%). ML direction predictions
(42-45% accuracy) aren't sufficient for edge. Stock strategy from Phase 2 remains the only
profitable approach (test 2023+ Sharpe 1.39).

## Project Structure

```
option_trade/
├── pipeline.py              # Data extraction & upload
├── research/
│   ├── option_analysis.py   # Feature engineering (Step 1)
│   ├── backtest.py          # Pinning distance analysis (Step 2)
│   ├── analysis.py          # RF feature importance (Step 3)
│   └── visualize.py         # Charts (Step 4)
├── features/
│   └── advanced_features.py # GEX, IV surface, OI dynamics (Step 9)
├── strategy/
│   ├── signals.py           # Signal generation (Step 5)
│   ├── pinning_strategy.py  # Strategy backtest engine (Step 6)
│   └── weekly_strategy.py   # Weekly options strategy (Step 11)
├── models/
│   └── predict.py           # LightGBM + XGBoost + PyTorch (Step 10)
├── rl/
│   ├── env.py               # Gymnasium RL environment (Step 7)
│   └── train.py             # PPO training & evaluation (Step 8)
├── data/                    # Parquet features, predictions, results
├── models/                  # Saved ML/RL models
├── output/                  # Raw OHLCV & options parquet
└── charts/                  # Visualization outputs
```

## Environment

- **Python**: 3.11 (conda env: snapshot-pipeline)
- **GPU**: NVIDIA RTX 2060 SUPER (CUDA 12.5)
- **PyTorch**: 2.6.0+cu124
- **ML**: lightgbm 4.6.0, xgboost 3.2.0, shap 0.51.0
- **RL**: gymnasium 1.2.3, stable-baselines3 2.8.0