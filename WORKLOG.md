# WORKLOG

## 2026-05-04

### Phase 2: Strategy Development & Backtest

**Step 5: `strategy/signals.py`**
- Dynamic pinning signal generator
- Confidence scoring: 40% volatility + 35% OI concentration + 25% target distance
- Short signals penalized 0.7x (direction bias from Phase 1)
- Three configs: conservative (atr≤1%, oi_conc≥5%), moderate (atr≤1.5%, oi_conc≥3%), aggressive (all tickers)
- Weekly signal mode: one trade per (ticker, expiration), enter at latest qualifying DTE

**Step 6: `strategy/pinning_strategy.py`**
- Full backtest engine supporting stock, vertical spread, butterfly, iron condor
- Stock: 5 bps slippage each side + 1 bps commission each side
- Options: actual bid/ask from options.parquet, 0.5% premium slippage, $0.65/contract commission
- Position sizing: confidence-scaled, equal, or Kelly
- Metrics: Sharpe, CAGR, win rate, profit factor, max drawdown, payoff ratio
- Train/val/test split: 2010-2019 / 2020-2022 / 2023+

**Stock Strategy Results (ETF, Moderate, Confidence-Scaled)**

| Period | Trades | Total Ret | Win Rate | Sharpe | Max DD | Profit Factor |
|--------|--------|-----------|----------|--------|--------|---------------|
| Train 2010-2019 | 790 | -6.9% | 48.2% | — | — | — |
| Val 2020-2022 | 247 | +15.3% | 50.6% | — | — | — |
| Test 2023+ | 715 | +46.5% | 53.1% | — | — | — |

**Stock Strategy Results (ETF, Conservative, Confidence-Scaled)**

| Period | Trades | Total Ret | CAGR | Win Rate | Sharpe | Max DD | PF |
|--------|--------|-----------|------|----------|--------|--------|-----|
| All | 652 | +10.3% | 0.8% | 50.5% | 0.31 | -7.0% | 1.13 |
| Train 2010-2019 | 386 | -3.7% | -0.5% | 48.7% | -0.19 | -7.0% | 0.92 |
| Val 2020-2022 | 72 | 0.0% | 0.0% | 48.6% | 0.02 | -4.1% | 1.01 |
| Test 2023+ | 194 | +14.6% | 3.7% | 54.6% | 1.39 | -2.7% | 1.71 |

**Options Strategy Findings**
- Vertical spreads (2% width, DTE 3-5): avg return -41%, win rate 16.6% — too wide for short DTE
- Butterfly spreads: similar poor performance
- Iron condor: structurally better for pinning thesis but needs optimization
- Conclusion: options strategies need RL to optimize strike selection and timing

**Dependencies Installed**
- torch 2.6.0+cu124 (GPU)
- gymnasium 1.2.3
- stable-baselines3 2.8.0
- scikit-learn, scipy, matplotlib

**Blocked → Resolved** (GPU now available: RTX 2060 SUPER)

### Phase 3: RL Optimization (Completed)

**Step 7: `rl/env.py`**
- Gymnasium environment: episode = one expiration cycle per (ticker, expiration)
- State (39 dim): 28 raw features + 9 ticker one-hot + position + unrealized PnL
- Features normalized: strikes → % distance from spot, OI/volume → log scale, DTE → /10
- Action: 3 discrete (short=0, flat=1, long=2)
- Reward: realized PnL (slippage 10bps round-trip) × reward_scaling + small unrealized incentive
- Penalty for no-trade episodes
- Train/val/test split: 2010-2019 / 2020-2022 / 2023+
- Episodes: train=1766, val=977, test=2029

**Step 8: `rl/train.py`**
- PPO (stable-baselines3), MLP [128, 128, 64]
- MetricsCallback: evaluate every 4096 steps on val set, save best model
- 5-seed ensemble training (200k steps each)
- Rule-based baseline comparison (max_volume_strike direction, ±0.5% threshold)

**RL Ensemble Results (5 seeds, test set)**

| Seed | Mean PnL | Win Rate | Sharpe | Trades/ep | Flat Rate |
|------|----------|----------|--------|-----------|-----------|
| 0 | -0.136% | 45.2% | -0.540 | 1.19 | 8.5% |
| 1 | -0.014% | 47.6% | -0.050 | 1.21 | 2.7% |
| 2 | -0.049% | 47.8% | -0.165 | 1.26 | 1.4% |
| 3 | -0.088% | 46.0% | -0.323 | 1.13 | 4.6% |
| 4 | -0.056% | 47.0% | -0.212 | 1.16 | 6.0% |
| **Ensemble** | **-0.068%** | **46.7%** | **-0.258** | 1.19 | 4.6% |
| **Baseline** | **-0.057%** | **34.2%** | **-0.246** | — | — |

**Key Findings**
- RL ensemble ≈ rule-based baseline (no significant improvement)
- Win rate improved (46.7% vs 34.2%) but PnL slightly worse due to more frequent smaller trades
- High variance across evaluation windows — learning curves unstable
- PPO struggles with sparse reward signal (most episodes only 3-5 steps)
- GPU: RTX 2060 SUPER available, but PPO MLP is CPU-bound (using device="cpu")

**Models saved**: `models/ppo_pinning_final_seed{0-4}.zip`, `models/ppo_pinning_best.zip`

### Next Steps

- Try different reward shaping: position-dependent reward, risk-adjusted reward (Sharpe per episode)
- Try different action space: continuous position sizing instead of discrete
- Use confidence-gated action: only trade when RL confidence > threshold (combine with Phase 2 signal)
- Try SAC or DQN instead of PPO
- Consider longer episode horizon: include pre-expiration week for context
- Alternative approach: use RL to optimize strike selection for options strategies (iron condor, butterfly)

---

## 2026-04-30

### Phase 1: Data Pipeline & Pinning Research

**Data Pipeline (`pipeline.py`)**
- Extract OHLCV (1999-2025) + options (2010-2026) from daily-snapshots zips (122GB)
- Streaming `unzip|awk` for fast options filtering
- Supplement with yfinance for recent data
- Upload 19 parquet files (~2GB) to Cloudflare R2

**Step 1: `research/option_analysis.py`**
- 8 pinning candidates: max_volume_strike, max_oi_strike, max_call_oi_strike, max_put_oi_strike, max_net_oi_strike, oi_mass_center, max_pain_strike, oi_top1_strike
- OI distribution features: concentration, skew, kurtosis, call_put_ratio
- Technical factors: SMA, RSI, ATR, realized vol, VWAP, support/resistance
- 34,213 rows → `data/features.parquet`

**Step 2: `research/backtest.py`**
- Per-expiration distance analysis, statistical tests vs random baseline
- Direction accuracy analysis
- `max_volume_strike` best indicator: 4.66% mean distance, 76.3% hit <2%

**Step 3: `research/analysis.py`**
- Random Forest feature importance (AUC=0.67 overall, 0.75 for ETFs)
- Top features: atr_pct, realized_vol_20d, days_to_expiry, atm_iv
- High-probability scenarios: SPY DTE=1, atr≤1%, oi_conc≥5% → 95.8% hit <2%

**Step 4: `research/visualize.py`**
- 6 charts: candidate comparison, ticker breakdown, DTE analysis, time stability, feature importance, factor combinations
