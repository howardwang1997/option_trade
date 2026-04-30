from __future__ import annotations

import os
import sys
import csv
import json
import zipfile
import io
import logging
import subprocess
import shutil
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import boto3
from botocore.config import Config
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

load_dotenv(Path(__file__).parent / ".env")

TARGET_TICKERS = ["NVDA", "GOOG", "GOOGL", "MSFT", "AMZN", "AAPL", "SPY", "QQQ", "GLD"]
TICKER_SET = set(TARGET_TICKERS)
REPO_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", "/home/howardwang/Downloads/daily-snapshots"))
OUTPUT_DIR = REPO_DIR / "output"
TEMP_DIR = REPO_DIR / "temp"

R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET = os.getenv("R2_BUCKET", "wall-street-baos")
R2_REGION = os.getenv("R2_REGION", "apac")

AWK_FILTER = "$2==\"NVDA\"||$2==\"GOOG\"||$2==\"GOOGL\"||$2==\"MSFT\"||$2==\"AMZN\"||$2==\"AAPL\"||$2==\"SPY\"||$2==\"QQQ\"||$2==\"GLD\""

OPTIONS_COLS = [
    "contract", "underlying", "expiration", "type", "strike", "style",
    "bid", "bid_size", "ask", "ask_size", "volume", "open_interest",
    "quote_date", "delta", "gamma", "theta", "vega", "implied_volatility",
]
OPTIONS_KEEP = [
    "quote_date", "underlying", "expiration", "type", "strike", "style",
    "bid", "bid_size", "ask", "ask_size", "volume", "open_interest",
    "delta", "gamma", "theta", "vega", "implied_volatility",
]
OPTIONS_KEEP_IDX = [OPTIONS_COLS.index(c) for c in OPTIONS_KEEP]


def get_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name=R2_REGION,
        config=Config(signature_version="s3v4"),
    )


# ---------------------------------------------------------------------------
# Phase 1: OHLCV extraction from repo zips (2002-2025)
# ---------------------------------------------------------------------------

def extract_ohlcv_from_zip(zpath: Path, ticker_dfs: dict[str, list[dict]]):
    try:
        with zipfile.ZipFile(zpath, "r") as zf:
            for name in zf.namelist():
                if not name.endswith("stocks.csv"):
                    continue
                date_str = name.replace("stocks.csv", "").strip()
                try:
                    datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    continue
                with zf.open(name) as f:
                    reader = csv.DictReader(io.TextIOWrapper(f, "utf-8"))
                    for row in reader:
                        sym = row.get("symbol", "")
                        if sym not in TICKER_SET:
                            continue
                        ticker_dfs[sym].append(
                            {
                                "date": date_str,
                                "open": float(row["open"]),
                                "high": float(row["high"]),
                                "low": float(row["low"]),
                                "close": float(row["close"]),
                                "volume": int(row["volume"]),
                                "source": "repo",
                            }
                        )
    except Exception as e:
        log.error(f"[OHLCV] Error processing {zpath}: {e}")


def run_phase1_ohlcv() -> dict[str, pd.DataFrame]:
    log.info("=== Phase 1: OHLCV extraction ===")
    ticker_dfs: dict[str, list[dict]] = {t: [] for t in TARGET_TICKERS}

    for zp in sorted(DATA_DIR.glob("20*.zip")):
        extract_ohlcv_from_zip(zp, ticker_dfs)

    for year_dir in sorted(DATA_DIR.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.startswith("20"):
            continue
        if year_dir.name in ("samples", "output", "2026", "temp"):
            continue
        for zp in sorted(year_dir.glob("*.zip")):
            extract_ohlcv_from_zip(zp, ticker_dfs)

    result = {}
    for ticker, rows in ticker_dfs.items():
        if not rows:
            log.warning(f"[OHLCV] No data for {ticker}")
            continue
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        df = df.reset_index(drop=True)
        result[ticker] = df
        log.info(f"[OHLCV] {ticker}: {len(df)} rows, {df['date'].min()} ~ {df['date'].max()}")

    return result


# ---------------------------------------------------------------------------
# Phase 2: Options extraction — streaming to disk via unzip | awk pipe
# ---------------------------------------------------------------------------

def extract_options_from_zip_streaming(zpath: Path):
    cmd = (
        f"unzip -p '{zpath}' '*options.csv' 2>/dev/null | "
        f"awk -F',' '{AWK_FILTER}'"
    )
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, bufsize=8 * 1024 * 1024)
    ticker_files = {}
    try:
        wrapper = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace")
        reader = csv.reader(wrapper)
        for cols in reader:
            if len(cols) < 18:
                continue
            underlying = cols[1]
            if underlying not in TICKER_SET:
                continue
            if underlying not in ticker_files:
                fpath = TEMP_DIR / f"options_{underlying}.csv"
                ticker_files[underlying] = open(fpath, "a", buffering=1 << 20)
            out_cols = [cols[i] for i in OPTIONS_KEEP_IDX]
            ticker_files[underlying].write(",".join(out_cols) + "\n")
    finally:
        for f in ticker_files.values():
            f.close()
        proc.wait(timeout=30)


def extract_options_2026_streaming(zpath: Path):
    try:
        with zipfile.ZipFile(zpath, "r") as zf:
            for name in zf.namelist():
                if not name.endswith(".txt"):
                    continue
                fname = Path(name).stem
                parts = fname.split("_")
                if len(parts) < 2:
                    continue
                ticker = parts[0]
                if ticker not in TICKER_SET:
                    continue
                out_path = TEMP_DIR / f"options_{ticker}.csv"
                with zf.open(name) as f, open(out_path, "a", buffering=1 << 20) as out:
                    wrapper = io.TextIOWrapper(f, "utf-8")
                    for line in wrapper:
                        cols = line.strip().split(",")
                        if len(cols) < 16:
                            continue
                        exp = cols[2]
                        typ = "call" if cols[3] == "c" else "put"
                        iv = cols[7] if cols[7] and cols[7] != "0" else cols[8]
                        row = ",".join([
                            cols[0], ticker, exp, typ, cols[1], "",
                            cols[5], "", cols[6], "",
                            cols[9], cols[10],
                            cols[11], cols[12], cols[14], cols[13], iv,
                        ])
                        out.write(row + "\n")
    except Exception as e:
        log.error(f"[Options 2026] Error: {e}")


def finalize_options() -> dict[str, pd.DataFrame]:
    result = {}
    for ticker in TARGET_TICKERS:
        fpath = TEMP_DIR / f"options_{ticker}.csv"
        if not fpath.exists():
            log.warning(f"[Options] No data for {ticker}")
            continue
        log.info(f"[Options] Loading {ticker} from temp CSV...")
        df = pd.read_csv(
            fpath,
            names=OPTIONS_KEEP,
            dtype={"volume": "Int64", "open_interest": "Int64", "bid_size": "Int64", "ask_size": "Int64"},
            na_values=["", "None"],
        )
        df["quote_date"] = pd.to_datetime(df["quote_date"])
        df["expiration"] = pd.to_datetime(df["expiration"])
        dedup = ["quote_date", "underlying", "expiration", "strike", "type"]
        df = df.sort_values("quote_date").drop_duplicates(subset=dedup, keep="last")
        df = df.reset_index(drop=True)
        result[ticker] = df
        log.info(f"[Options] {ticker}: {len(df)} rows, {df['quote_date'].min()} ~ {df['quote_date'].max()}")
    return result


def run_phase2_options() -> dict[str, pd.DataFrame]:
    log.info("=== Phase 2: Options extraction (2010-2025) ===")
    TEMP_DIR.mkdir(exist_ok=True)

    for ticker in TARGET_TICKERS:
        fpath = TEMP_DIR / f"options_{ticker}.csv"
        if fpath.exists():
            fpath.unlink()

    zip_list = []
    for year_dir in sorted(DATA_DIR.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.startswith("20"):
            continue
        if year_dir.name in ("samples", "output", "2026", "temp"):
            continue
        for zp in sorted(year_dir.glob("*.zip")):
            zip_list.append(zp)

    t_start = time.time()
    for i, zp in enumerate(zip_list):
        t0 = time.time()
        extract_options_from_zip_streaming(zp)
        log.info(f"[Options] [{i+1}/{len(zip_list)}] {zp.relative_to(DATA_DIR)} {time.time()-t0:.1f}s")

    log.info("=== Phase 3: Options extraction (2026) ===")
    dir_2026 = DATA_DIR / "2026"
    if dir_2026.exists():
        for zp in sorted(dir_2026.glob("*.zip")):
            extract_options_2026_streaming(zp)

    log.info("[Options] Finalizing (loading + dedup)...")
    result = finalize_options()
    elapsed = time.time() - t_start
    log.info(f"[Options] Total time: {elapsed:.0f}s")
    return result


# ---------------------------------------------------------------------------
# Phase 4: yfinance OHLCV supplement
# ---------------------------------------------------------------------------

def run_phase4_yfinance(ohlcv_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    log.info("=== Phase 4: yfinance supplement ===")
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed, skipping supplement")
        return ohlcv_data

    for ticker in TARGET_TICKERS:
        log.info(f"[yfinance] Downloading {ticker}")
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="max", auto_adjust=True)
            if hist.empty:
                log.warning(f"[yfinance] No data for {ticker}")
                continue
            hist = hist.reset_index()
            hist.columns = [c.lower().replace(" ", "_") for c in hist.columns]
            yf_df = hist.iloc[:, :6].copy()
            yf_df.columns = ["date", "open", "high", "low", "close", "volume"]
            yf_df["source"] = "yfinance"
            yf_df["date"] = pd.to_datetime(yf_df["date"]).dt.tz_localize(None)

            if ticker in ohlcv_data:
                repo_df = ohlcv_data[ticker]
                repo_dates = set(repo_df["date"].dt.strftime("%Y-%m-%d"))
                yf_new = yf_df[~yf_df["date"].dt.strftime("%Y-%m-%d").isin(repo_dates)]
                if not yf_new.empty:
                    combined = pd.concat([repo_df, yf_new], ignore_index=True)
                    combined = combined.sort_values("date").drop_duplicates(
                        subset=["date"], keep="first"
                    ).reset_index(drop=True)
                    ohlcv_data[ticker] = combined
                    log.info(f"[yfinance] {ticker}: added {len(yf_new)} supplementary rows")
                else:
                    log.info(f"[yfinance] {ticker}: no missing dates to fill")
            else:
                ohlcv_data[ticker] = yf_df
                log.info(f"[yfinance] {ticker}: using yfinance as primary source ({len(yf_df)} rows)")
        except Exception as e:
            log.error(f"[yfinance] Error downloading {ticker}: {e}")

    return ohlcv_data


# ---------------------------------------------------------------------------
# Phase 5: Parquet output + R2 upload
# ---------------------------------------------------------------------------

def write_parquet_files(
    ohlcv_data: dict[str, pd.DataFrame],
    options_data: dict[str, pd.DataFrame],
):
    log.info("=== Phase 5: Writing parquet files ===")
    OUTPUT_DIR.mkdir(exist_ok=True)
    manifest = {"tickers": {}, "generated_at": datetime.utcnow().isoformat()}

    for ticker in TARGET_TICKERS:
        tdir = OUTPUT_DIR / ticker
        tdir.mkdir(exist_ok=True)
        entry = {"ticker": ticker}

        if ticker in ohlcv_data:
            df = ohlcv_data[ticker]
            path = tdir / "ohlcv.parquet"
            df.to_parquet(path, index=False, engine="pyarrow")
            entry["ohlcv"] = {
                "rows": len(df),
                "date_range": [str(df["date"].min()), str(df["date"].max())],
                "file_size_bytes": path.stat().st_size,
            }
            log.info(f"[Parquet] {ticker}/ohlcv.parquet: {len(df)} rows, {path.stat().st_size / 1024:.1f} KB")

        if ticker in options_data:
            df = options_data[ticker]
            path = tdir / "options.parquet"
            df.to_parquet(path, index=False, engine="pyarrow")
            entry["options"] = {
                "rows": len(df),
                "date_range": [str(df["quote_date"].min()), str(df["quote_date"].max())],
                "file_size_bytes": path.stat().st_size,
            }
            log.info(f"[Parquet] {ticker}/options.parquet: {len(df)} rows, {path.stat().st_size / 1024:.1f} KB")

        manifest["tickers"][ticker] = entry

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"[Manifest] Written to {manifest_path}")

    return manifest


def upload_to_r2():
    log.info("=== Uploading to R2 ===")
    client = get_r2_client()

    for ticker_dir in sorted(OUTPUT_DIR.iterdir()):
        if not ticker_dir.is_dir():
            continue
        for fpath in sorted(ticker_dir.iterdir()):
            if not fpath.is_file():
                continue
            key = f"data/{ticker_dir.name}/{fpath.name}"
            log.info(f"[R2] Uploading {key} ({fpath.stat().st_size / 1024 / 1024:.1f} MB)")
            try:
                client.upload_file(str(fpath), R2_BUCKET, key)
                log.info(f"[R2] Uploaded {key}")
            except Exception as e:
                log.error(f"[R2] Failed to upload {key}: {e}")

    manifest_path = OUTPUT_DIR / "manifest.json"
    if manifest_path.exists():
        key = "data/manifest.json"
        log.info(f"[R2] Uploading {key}")
        try:
            client.upload_file(str(manifest_path), R2_BUCKET, key)
            log.info(f"[R2] Uploaded {key}")
        except Exception as e:
            log.error(f"[R2] Failed to upload {key}: {e}")

    log.info("=== R2 upload complete ===")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_temp():
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        log.info("[Cleanup] Removed temp directory")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    phase = os.getenv("PHASE", "all")
    t_total = time.time()

    ohlcv_data = {}
    options_data = {}

    if phase in ("all", "ohlcv"):
        ohlcv_data = run_phase1_ohlcv()

    if phase in ("all", "options"):
        options_data = run_phase2_options()

    if phase in ("all", "yfinance"):
        ohlcv_data = run_phase4_yfinance(ohlcv_data)

    if phase in ("all", "write"):
        write_parquet_files(ohlcv_data, options_data)

    if phase in ("all", "upload"):
        upload_to_r2()

    cleanup_temp()
    log.info(f"=== Done in {(time.time() - t_total):.0f}s ===")


if __name__ == "__main__":
    main()
