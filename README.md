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