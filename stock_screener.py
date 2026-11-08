#!/usr/bin/env python3
"""Lightweight stock screener with a 1-10 rating, disk cache, flags, HTML report and backtest.

Built for low-power ARM machines: batched downloads, incremental disk cache, float32
frames, vectorized pandas math, rule-based VADER sentiment (no deep-learning models).

Install:  pip install yfinance pandas numpy nltk lxml

Usage:
  python stock-screener.py                                  # default 4-ticker watchlist
  python stock-screener.py --sp500 --html                   # S&P 500 + HTML report
  python stock-screener.py --universe sp500 nasdaq100 dow30
  python stock-screener.py --file my_tickers.txt --tickers TSLA BTC-USD VOD.L
  python stock-screener.py --backtest                       # test the rating's track record
  python stock-screener.py --sp500 --no-turnaround          # skip the Hammer turnaround list
  python stock-screener.py --backtest                       # rating + turnaround backtest
  python stock-screener.py --sp500 --weights-file screener_output/suggested_weights.json
  python stock-screener.py --sp500 --quiet                  # for cron / scheduled runs
  python stock-screener.py --from-csv screener_output/screener_results_2026-11-08.csv
                                                            # re-rank a saved CSV (no downloads)

The rating is a rule-based screening heuristic, NOT a price forecast.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
import time
import warnings
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_WATCHLIST = ["AAPL", "MSFT", "GOOGL", "AMZN"]
# 6 months (~126 trading days) cannot produce a 200-day SMA; 1 year is the minimum.
DEFAULT_PERIOD = "1y"
RSI_PERIOD = 14
SMA_FAST = 50
SMA_SLOW = 200
RSI_ENTRY_THRESHOLD = 40.0
MAX_HEADLINES = 10
OHLC_COLS = ["Open", "High", "Low", "Close"]

RS_LOOKBACK = 63          # ~3 months of trading days for relative strength vs SPY
HIGH_LOOKBACK = 252       # ~52 weeks
BENCHMARK = "SPY"

DEFAULT_CHUNK = 40
DEFAULT_TOP_DETAIL = 25   # news sentiment + earnings dates are fetched for the top N only
DEFAULT_SHOW = 25
DEFAULT_MIN_DOLLAR_VOL = 5_000_000.0
HIGH_VOL_ATR_PCT = 5.0
EARNINGS_WARN_DAYS = 7
DEFAULT_STOP_ATR = 1.5
DEFAULT_RISK_DOLLARS = 100.0
DEFAULT_COST_BPS = 10.0   # round-trip trading cost assumed in backtests
QUIET = False

CACHE_DIR_DEFAULT = ".screener_cache"
OUTPUT_DIR_DEFAULT = "screener_output"
CACHE_KEEP_ROWS = 400
CACHE_TOLERANCE = 0.003   # overlap mismatch above this triggers a full re-download

# Rating weights (sum to 1). Tune freely; run --backtest to see the effect.
RATING_WEIGHTS = {
    "trend": 0.30,
    "rsi": 0.20,
    "rel_strength": 0.15,
    "macd": 0.10,
    "sentiment": 0.15,
    "candle": 0.05,
    "near_high": 0.05,
}
_RSI_X = [20.0, 40.0, 50.0, 60.0, 80.0]   # RSI -> bullishness (oversold is favorable)
_RSI_Y = [1.00, 0.80, 0.50, 0.35, 0.00]
# Weights actually used for rating. --weights-file replaces these at runtime.
ACTIVE_WEIGHTS: dict[str, float] = dict(RATING_WEIGHTS)

INDEXES: dict[str, tuple[str, int]] = {
    # name: (Wikipedia page, minimum rows expected in the constituents table)
    "sp500": ("List_of_S%26P_500_companies", 400),
    "sp400": ("List_of_S%26P_400_companies", 300),
    "sp600": ("List_of_S%26P_600_companies", 500),
    "nasdaq100": ("Nasdaq-100", 90),
    "dow30": ("Dow_Jones_Industrial_Average", 25),
}


def resolve_output(user_path: Optional[str], default_name: str, out_dir: str) -> Path:
    """Place output files in out_dir. A bare filename goes inside the folder; a path that
    already names a directory (e.g. results/x.csv or /tmp/x.csv) is respected as given."""
    if user_path:
        p = Path(user_path)
        if p.parent == Path("."):
            p = Path(out_dir) / p.name
    else:
        p = Path(out_dir) / default_name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def log(msg: str) -> None:
    if QUIET and msg.startswith("[info]"):
        return
    print(msg, file=sys.stderr, flush=True)


def _f(x: Any) -> float:
    """Safe native-float cast; NaN for anything non-finite or non-numeric."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #
def _now_et() -> datetime:
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return datetime.now()


def today_et() -> date:
    return _now_et().date()


def expected_last_trading_day() -> date:
    """Most recent US trading day whose daily bar should be complete (after ~16:30 ET)."""
    ref = (_now_et() - timedelta(hours=16, minutes=30)).date()
    while ref.weekday() >= 5:
        ref -= timedelta(days=1)
    return ref


# --------------------------------------------------------------------------- #
# Ticker universe
# --------------------------------------------------------------------------- #
def _normalize_symbol(sym: Any, from_wikipedia: bool = False) -> str:
    """Uppercase/strip a symbol. Wikipedia's BRK.B becomes Yahoo's BRK-B.

    User-typed symbols are left alone so foreign suffixes like VOD.L or SHOP.TO survive.
    """
    s = str(sym).strip().upper()
    return s.replace(".", "-") if from_wikipedia else s


DOW30_FALLBACK = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
]


def _symbol_series(tbl: pd.DataFrame, min_rows: int) -> Optional[pd.Series]:
    """Find a ticker column in a table, tolerating MultiIndex headers and footnote marks."""
    if len(tbl) < min_rows:
        return None
    for i, col in enumerate(tbl.columns):
        name = " ".join(map(str, col)) if isinstance(col, tuple) else str(col)
        name = re.sub(r"\[.*?\]", "", name).strip().lower()
        if (name in ("symbol", "ticker") or name.endswith(" symbol")
                or name.endswith(" ticker") or "ticker symbol" in name):
            return tbl.iloc[:, i]
    return None


def get_index_tickers(name: str) -> list[str]:
    """Fetch an index's constituents from Wikipedia, with a local CSV cache fallback."""
    page, min_rows = INDEXES[name]
    cache = Path(CACHE_DIR_DEFAULT) / f"{name}_tickers.csv"
    try:
        import requests

        resp = requests.get(
            f"https://en.wikipedia.org/wiki/{page}",
            headers={"User-Agent": "Mozilla/5.0 (stock-screener)"},
            timeout=20,
        )
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        for tbl in tables:
            col = _symbol_series(tbl, min_rows)
            if col is None:
                continue
            syms = [_normalize_symbol(s, True) for s in col.dropna().tolist()]
            syms = [t for t in dict.fromkeys(syms) if t and re.fullmatch(r"[A-Z0-9.\-]{1,10}", t)]
            if len(syms) >= min_rows:
                try:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    pd.Series(syms, name="Symbol").to_csv(cache, index=False)
                except OSError:
                    pass
                return syms
        seen = [f"{len(t)} rows: {[str(c) for c in t.columns][:4]}"
                for t in tables if len(t) >= 10][:4]
        raise ValueError(f"constituents table not found; tables seen: {seen}")
    except Exception as exc:
        log(f"[warn] could not fetch {name} list online ({exc})")

    if cache.exists():
        try:
            cached = pd.read_csv(cache)["Symbol"].dropna().tolist()
            syms = list(dict.fromkeys(_normalize_symbol(s) for s in cached))
            if syms:
                log(f"[info] using cached {name} list from {cache} ({len(syms)} symbols)")
                return syms
        except Exception as exc:
            log(f"[warn] {name} cache unreadable ({exc})")

    if name == "dow30":
        log("[warn] using a built-in Dow 30 list; membership may be slightly out of date")
        return list(DOW30_FALLBACK)

    log(f"[warn] no {name} tickers available (try: pip install lxml)")
    return []


def load_ticker_file(path: str) -> list[str]:
    """Read tickers from a text/CSV file: separated by newlines, commas or spaces."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        log(f"[warn] cannot read ticker file {path} ({exc})")
        return []
    out: list[str] = []
    for tok in re.split(r"[,\s;]+", text):
        tok = tok.strip().strip('"').strip("'")
        if (tok and tok.upper() not in ("SYMBOL", "TICKER")
                and re.fullmatch(r"[A-Za-z0-9.\-^=]{1,12}", tok)):
            out.append(_normalize_symbol(tok))
    return list(dict.fromkeys(out))


# --------------------------------------------------------------------------- #
# Data retrieval (batched, retried, cached)
# --------------------------------------------------------------------------- #
def _clean_ohlc(sub: Any) -> pd.DataFrame:
    """Normalize one ticker's raw frame to a lean float32 OHLCV DataFrame."""
    if not isinstance(sub, pd.DataFrame) or sub.empty:
        return pd.DataFrame()
    df = sub.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    if not set(OHLC_COLS).issubset(df.columns):
        return pd.DataFrame()

    cols = OHLC_COLS + (["Volume"] if "Volume" in df.columns else [])
    df = df[cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=OHLC_COLS)
    if df.empty:
        return pd.DataFrame()
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df["Volume"] = df["Volume"].fillna(0.0)
    df = df.astype(np.float32)

    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _extract_ticker(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Pull one ticker out of a batch download, whatever the column layout is."""
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw
    lv0 = raw.columns.get_level_values(0)
    lv1 = raw.columns.get_level_values(1)
    if ticker in lv0:
        return raw[ticker]
    if ticker in lv1:
        return raw.xs(ticker, axis=1, level=1)
    return pd.DataFrame()


def download_chunk(
    tickers: list[str],
    period: Optional[str] = None,
    start: Optional[str] = None,
    retries: int = 3,
) -> dict[str, pd.DataFrame]:
    """Download a batch with retry/backoff; returns {ticker: clean OHLCV frame}."""
    if not tickers:
        return {}
    delays = [2.0, 5.0, 10.0]
    raw: Any = None
    for attempt in range(max(1, retries)):
        try:
            kwargs: dict[str, Any] = dict(
                interval="1d", auto_adjust=True, group_by="ticker",
                progress=False, threads=True,
            )
            if start:
                kwargs["start"] = start
            else:
                kwargs["period"] = period or DEFAULT_PERIOD
            raw = yf.download(tickers, **kwargs)
        except Exception as exc:
            log(f"[warn] download attempt {attempt + 1}/{retries} failed ({exc})")
            raw = None
        if isinstance(raw, pd.DataFrame) and not raw.empty:
            break
        if attempt < retries - 1:
            time.sleep(delays[min(attempt, len(delays) - 1)])

    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return {}

    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = _clean_ohlc(_extract_ticker(raw, t))
        except Exception:
            continue
        if not df.empty:
            out[t] = df
    return out


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / (re.sub(r"[^A-Za-z0-9_.-]", "_", ticker) + ".csv.gz")


def read_cache(cache_dir: Path, ticker: str) -> Optional[pd.DataFrame]:
    path = _cache_path(cache_dir, ticker)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = _clean_ohlc(df)
        return df if not df.empty else None
    except Exception:
        return None


def write_cache(cache_dir: Path, ticker: str, df: pd.DataFrame) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.tail(CACHE_KEEP_ROWS).to_csv(_cache_path(cache_dir, ticker), compression="gzip")
    except Exception as exc:
        log(f"[warn] {ticker}: could not write cache ({exc})")


def merge_prices(cached: pd.DataFrame, fresh: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Merge new bars into cached ones. ok=False if adjusted history no longer lines up."""
    overlap = cached.index.intersection(fresh.index)
    if len(overlap) == 0:
        return cached, False
    a = cached.loc[overlap, "Close"].astype(np.float64)
    b = fresh.loc[overlap, "Close"].astype(np.float64)
    rel = ((a - b).abs() / b.abs().replace(0, np.nan)).max()
    if not np.isfinite(rel) or rel > CACHE_TOLERANCE:
        return cached, False
    merged = pd.concat([cached[~cached.index.isin(fresh.index)], fresh]).sort_index()
    return merged.tail(CACHE_KEEP_ROWS), True


def _completed_only(df: pd.DataFrame, expected: date) -> pd.DataFrame:
    """Drop in-progress bars so every run analyzes the last COMPLETED close."""
    return df[df.index <= pd.Timestamp(expected)]


def get_frames(
    tickers: list[str],
    period: str,
    cache_dir: Optional[Path],
    refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], bool]:
    """Return ({ticker: frame}, hit_network). Uses/updates the disk cache when enabled."""
    frames: dict[str, pd.DataFrame] = {}
    expected = expected_last_trading_day()
    need_full: list[str] = []
    need_inc: dict[str, pd.DataFrame] = {}
    used_network = False

    for t in tickers:
        cached = None
        if cache_dir is not None and not refresh:
            cached = read_cache(cache_dir, t)
        if cached is None or cached.empty:
            need_full.append(t)
        elif cached.index[-1].date() >= expected:
            frames[t] = _completed_only(cached, expected)
        else:
            need_inc[t] = cached

    if need_full:
        used_network = True
        for t, df in download_chunk(need_full, period=period).items():
            df = _completed_only(df, expected)
            if df.empty:
                continue
            frames[t] = df
            if cache_dir is not None:
                write_cache(cache_dir, t, df)

    if need_inc:
        used_network = True
        start = (min(c.index[-1] for c in need_inc.values()) - pd.Timedelta(days=7))
        new = download_chunk(list(need_inc), start=start.date().isoformat(), retries=1)
        for t, cached in need_inc.items():
            fresh = new.get(t)
            if fresh is None or fresh.empty:
                frames[t] = _completed_only(cached, expected)  # keep stale cache
                continue
            merged, ok = merge_prices(cached, _completed_only(fresh, expected))
            if not ok:  # split/dividend re-adjusted history: refresh this ticker fully
                full = download_chunk([t], period=period, retries=2).get(t)
                merged = _completed_only(full, expected) if full is not None else cached
            frames[t] = merged
            if cache_dir is not None and not merged.empty:
                write_cache(cache_dir, t, merged)

    return {t: df for t, df in frames.items() if not df.empty}, used_network


# --------------------------------------------------------------------------- #
# Features: indicators, candlesticks
# --------------------------------------------------------------------------- #
def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI, fully vectorized."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def candlestick_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized candlestick pattern booleans for every bar (causal: uses only past bars)."""
    o, h, l, c = (df[x].astype(np.float64) for x in OHLC_COLS)
    po, pc = o.shift(1), c.shift(1)
    body = (c - o).abs()
    rng = h - l
    top = pd.concat([o, c], axis=1).max(axis=1)
    bottom = pd.concat([o, c], axis=1).min(axis=1)
    upper = h - top
    lower = bottom - l
    valid = rng > 0
    return pd.DataFrame({
        "Doji": valid & (body <= 0.1 * rng),
        "Hammer": valid & (body > 0) & (lower >= 2 * body) & (upper <= 0.3 * rng),
        "Shooting Star": valid & (body > 0) & (upper >= 2 * body) & (lower <= 0.3 * rng),
        "Bullish Engulfing": (pc < po) & (c > o) & (c >= po) & (o <= pc),
        "Bearish Engulfing": (pc > po) & (c < o) & (c <= po) & (o >= pc),
    }, index=df.index)


def detect_candlestick_patterns(df: pd.DataFrame) -> str:
    """Comma-separated patterns on the most recent bar."""
    if len(df) < 2:
        return "n/a"
    last = candlestick_flags(df.tail(3)).iloc[-1]
    found = [name for name, v in last.items() if bool(v)]
    return ", ".join(found) if found else "None"


def compute_features(df: pd.DataFrame, spy_close: Optional[pd.Series] = None) -> pd.DataFrame:
    """Append indicator columns to a copy of an OHLCV frame (all causal, no look-ahead)."""
    feat = df.copy()
    close = feat["Close"]
    high, low = feat["High"], feat["Low"]

    feat["RSI14"] = compute_rsi(close)
    feat["SMA50"] = close.rolling(SMA_FAST, min_periods=SMA_FAST).mean()
    feat["SMA200"] = close.rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    feat["MACD"] = ema12 - ema26
    feat["MACDsig"] = feat["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()],
                   axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    feat["ATRpct"] = atr / close * 100.0

    high52 = high.rolling(HIGH_LOOKBACK, min_periods=60).max()
    feat["PctFromHigh"] = (close / high52 - 1.0) * 100.0

    if spy_close is not None and len(spy_close) > RS_LOOKBACK:
        spy = spy_close.astype(np.float64).reindex(feat.index, method="ffill")
        stock_ret = close.astype(np.float64) / close.astype(np.float64).shift(RS_LOOKBACK) - 1.0
        spy_ret = spy / spy.shift(RS_LOOKBACK) - 1.0
        feat["RS63"] = (stock_ret - spy_ret) * 100.0
    else:
        feat["RS63"] = np.nan

    vol = feat["Volume"].astype(np.float64)
    feat["AvgDollarVol"] = vol.rolling(20, min_periods=10).mean() * close.astype(np.float64)

    flags = candlestick_flags(feat)
    feat["CandleBull"] = (flags["Hammer"] | flags["Bullish Engulfing"]).astype(np.float32)
    feat["CandleBear"] = (flags["Shooting Star"] | flags["Bearish Engulfing"]).astype(np.float32)
    return feat


# --------------------------------------------------------------------------- #
# Rating (1 = most bearish setup, 10 = most bullish setup)
# --------------------------------------------------------------------------- #
def load_weights(path: str) -> dict[str, float]:
    """Load signed component weights from a JSON file (e.g. one written by --backtest).

    A negative weight means the OPPOSITE of that signal scored well, so it is applied inverted.
    """
    import json

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("weights file must contain a non-empty JSON object")
    weights: dict[str, float] = {}
    for key, val in raw.items():
        if key not in RATING_WEIGHTS:
            raise ValueError(f"unknown weight '{key}' (valid: {sorted(RATING_WEIGHTS)})")
        w = _f(val)
        if not np.isfinite(w):
            raise ValueError(f"weight for '{key}' is not a number")
        weights[key] = w
    if not any(w != 0.0 for w in weights.values()):
        raise ValueError("all weights are zero")
    return weights


def rating_components(feat: pd.DataFrame, use_sentiment: bool = True) -> dict[str, pd.Series]:
    """Each rating ingredient as a 0..1 bullishness score (0.5 = neutral/unknown)."""
    def num(col: str) -> pd.Series:
        return pd.to_numeric(feat[col], errors="coerce")

    idx = feat.index
    close, s50, s200 = num("Close"), num("SMA50"), num("SMA200")

    p1 = (close > s50).astype(float).where(s50.notna(), 0.5)
    p2 = (close > s200).astype(float).where(s200.notna(), 0.5)
    p3 = (s50 > s200).astype(float).where(s50.notna() & s200.notna(), 0.5)
    trend = 0.5 * p1 + 0.3 * p2 + 0.2 * p3

    rsi = num("RSI14")
    rsi_s = pd.Series(np.interp(rsi.fillna(50.0).to_numpy(float), _RSI_X, _RSI_Y), index=idx)

    rs = num("RS63")
    rs_s = pd.Series(np.interp(rs.fillna(0.0).to_numpy(float), [-20.0, 0.0, 20.0],
                               [0.0, 0.5, 1.0]), index=idx).where(rs.notna(), 0.5)

    m, sg = num("MACD"), num("MACDsig")
    macd_s = (0.5 * (m > sg).astype(float) + 0.5 * (m > 0).astype(float)
              ).where(m.notna() & sg.notna(), 0.5)

    ph = num("PctFromHigh")
    high_s = pd.Series(np.interp(ph.fillna(-15.0).to_numpy(float), [-30.0, 0.0], [0.0, 1.0]),
                       index=idx).where(ph.notna(), 0.5)

    candle = (0.5 + 0.5 * num("CandleBull").fillna(0.0)
              - 0.5 * num("CandleBear").fillna(0.0)).clip(0.0, 1.0)

    parts = {
        "trend": trend, "rsi": rsi_s, "rel_strength": rs_s, "macd": macd_s,
        "candle": candle, "near_high": high_s,
    }
    if use_sentiment and "Sentiment" in feat.columns:
        sent = num("Sentiment").fillna(0.0).clip(-1.0, 1.0)
        parts["sentiment"] = (sent + 1.0) / 2.0
    return parts


def combine_components(parts: dict[str, pd.Series],
                       weights: Optional[dict[str, float]] = None) -> pd.Series:
    """Blend component scores into a 1-10 rating. Negative weights invert a component."""
    w = ACTIVE_WEIGHTS if weights is None else weights
    keys = [k for k in parts if w.get(k, 0.0) != 0.0]
    any_part = next(iter(parts.values()))
    if not keys:
        return pd.Series(5.5, index=any_part.index)
    total = sum(abs(w[k]) for k in keys)
    comp = sum(abs(w[k]) * (parts[k] if w[k] > 0 else 1.0 - parts[k]) for k in keys) / total
    return (1.0 + 9.0 * comp).clip(1.0, 10.0).round(1)


def rating_from_features(feat: pd.DataFrame, use_sentiment: bool = True,
                         weights: Optional[dict[str, float]] = None) -> pd.Series:
    """Heuristic 1-10 rating (10 = most bullish setup). A screening score, not a forecast."""
    return combine_components(rating_components(feat, use_sentiment), weights)


# --------------------------------------------------------------------------- #
# Sentiment (NLTK VADER) and earnings dates
# --------------------------------------------------------------------------- #
_VADER: Optional[Any] = None
_VADER_TRIED = False


def _get_vader() -> Optional[Any]:
    """Lazy-load VADER once; return None if unavailable."""
    global _VADER, _VADER_TRIED
    if _VADER_TRIED:
        return _VADER
    _VADER_TRIED = True
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        try:
            _VADER = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            _VADER = SentimentIntensityAnalyzer()
    except Exception as exc:
        log(f"[warn] VADER unavailable ({exc}); sentiment set to 0.0")
        _VADER = None
    return _VADER


def fetch_headlines(ticker: str, limit: int = MAX_HEADLINES) -> list[str]:
    """Recent headlines, tolerating both old and new yfinance news schemas."""
    try:
        items = yf.Ticker(ticker).news
    except Exception:
        return []
    if not isinstance(items, list):
        return []
    titles: list[str] = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        if not title and isinstance(item.get("content"), dict):
            title = item["content"].get("title")
        if isinstance(title, str) and title.strip():
            titles.append(title.strip())
    return titles


def sentiment_score(headlines: list[str]) -> float:
    """Mean VADER compound score in [-1, 1]; 0.0 if no data."""
    if not headlines:
        return 0.0
    vader = _get_vader()
    if vader is None:
        return 0.0
    scores = [float(vader.polarity_scores(h)["compound"]) for h in headlines]
    return float(np.mean(scores)) if scores else 0.0


def fetch_next_earnings_days(ticker: str) -> Optional[int]:
    """Days until the next earnings date, or None if unknown/not applicable."""
    try:
        cal = yf.Ticker(ticker).calendar
    except Exception:
        return None
    raw: Any = None
    if isinstance(cal, dict):
        raw = cal.get("Earnings Date")
    elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
        raw = cal.loc["Earnings Date"].tolist()
    if raw is None:
        return None
    items = list(raw) if isinstance(raw, (list, tuple, np.ndarray, pd.Series)) else [raw]
    today = today_et()
    future: list[date] = []
    for x in items:
        try:
            d = pd.Timestamp(x).date()
        except Exception:
            continue
        if d >= today:
            future.append(d)
    return int((min(future) - today).days) if future else None


# --------------------------------------------------------------------------- #
# Per-ticker analysis and ranking
# --------------------------------------------------------------------------- #
def hammer_confirmation(df: pd.DataFrame) -> str:
    """"Pending" = hammer on the latest bar; "Confirmed" = previous bar was a hammer and the
    latest bar closed above its high; "" otherwise."""
    if len(df) < 3:
        return ""
    tail = df.tail(3)
    fl = candlestick_flags(tail)
    if bool(fl["Hammer"].iloc[-1]):
        return "Pending"
    if bool(fl["Hammer"].iloc[-2]) and float(tail["Close"].iloc[-1]) > float(tail["High"].iloc[-2]):
        return "Confirmed"
    return ""


def analyze_frame(ticker: str, df: pd.DataFrame,
                  spy_close: Optional[pd.Series]) -> Optional[dict[str, Any]]:
    """Latest-bar features + original 0-2 score for one ticker (no network calls)."""
    if df.empty or len(df) < RSI_PERIOD + 1:
        return None
    feat = compute_features(df, spy_close)
    last = feat.iloc[-1]

    close, rsi = _f(last["Close"]), _f(last["RSI14"])
    sma50, sma200 = _f(last["SMA50"]), _f(last["SMA200"])
    dollar_vol = _f(last["AvgDollarVol"])
    if not np.isfinite(dollar_vol) or dollar_vol <= 0:
        dollar_vol = float("nan")  # unknown (e.g. indexes report no volume)

    # NaN comparisons are False, so missing indicators score 0 safely.
    above_sma50 = bool(np.isfinite(sma50) and close > sma50)
    rsi_dip = bool(np.isfinite(rsi) and rsi < RSI_ENTRY_THRESHOLD)

    return {
        "Ticker": ticker,
        "Score": int(above_sma50) + int(rsi_dip),
        "Close": close,
        "RSI14": rsi,
        "SMA50": sma50,
        "SMA200": sma200,
        "MACD": _f(last["MACD"]),
        "MACDsig": _f(last["MACDsig"]),
        "RS63": _f(last["RS63"]),
        "PctFromHigh": _f(last["PctFromHigh"]),
        "ATRpct": _f(last["ATRpct"]),
        "AvgDollarVol": dollar_vol,
        "CandleBull": _f(last["CandleBull"]),
        "CandleBear": _f(last["CandleBear"]),
        "Above SMA50": above_sma50,
        "RSI<40": rsi_dip,
        "Sentiment": float("nan"),
        "DaysToEarnings": float("nan"),
        "Pattern": detect_candlestick_patterns(df),
        "HammerConfirm": hammer_confirmation(df),
    }


def build_flags(out: pd.DataFrame, min_dollar_vol: float) -> pd.Series:
    flags: list[str] = []
    for dv, atr, days in zip(out["AvgDollarVol"], out["ATRpct"], out["DaysToEarnings"]):
        f: list[str] = []
        if pd.notna(dv) and dv < min_dollar_vol:
            f.append("LowLiquidity")
        if pd.notna(atr) and atr >= HIGH_VOL_ATR_PCT:
            f.append("HighVolatility")
        if pd.notna(days) and 0 <= days <= EARNINGS_WARN_DAYS:
            f.append(f"Earnings in {int(days)}d")
        flags.append(", ".join(f))
    return pd.Series(flags, index=out.index, dtype=object)


def load_benchmark(period: str, cache_dir: Optional[Path], refresh: bool) -> Optional[pd.Series]:
    frames, _ = get_frames([BENCHMARK], period, cache_dir, refresh)
    df = frames.get(BENCHMARK)
    if df is None or df.empty:
        log(f"[warn] could not load {BENCHMARK}; relative strength will count as neutral")
        return None
    return df["Close"]


def build_ranking(
    tickers: list[str],
    period: str,
    chunk_size: int = DEFAULT_CHUNK,
    top_detail: int = DEFAULT_TOP_DETAIL,
    cache_dir: Optional[Path] = None,
    refresh: bool = False,
    min_dollar_vol: float = DEFAULT_MIN_DOLLAR_VOL,
) -> pd.DataFrame:
    spy_close = load_benchmark(period, cache_dir, refresh)
    rows: list[dict[str, Any]] = []
    total = len(tickers)

    for start in range(0, total, chunk_size):
        chunk = tickers[start:start + chunk_size]
        log(f"[info] loading {start + 1}-{start + len(chunk)} of {total} ...")
        frames, used_network = get_frames(chunk, period, cache_dir, refresh)
        for t, df in frames.items():
            try:
                result = analyze_frame(t, df, spy_close)
            except Exception as exc:  # never let one ticker kill the run
                log(f"[warn] {t}: analysis failed ({exc})")
                continue
            if result is not None:
                rows.append(result)
        missing = [t for t in chunk if t not in frames]
        if missing:
            log(f"[warn] no usable data for: {', '.join(missing)}")
        del frames
        if used_network and start + chunk_size < total:
            time.sleep(1.0)  # be polite to the API

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["Rating"] = rating_from_features(out)

    # News sentiment + earnings dates only for the top-N candidates (pre-sentiment rating).
    if top_detail > 0:
        order = out.sort_values(["Rating", "RSI14"], ascending=[False, True],
                                na_position="last", kind="mergesort").index
        candidates = list(order[:top_detail])
        log(f"[info] fetching news and earnings dates for top {len(candidates)} ...")
        for idx in candidates:
            sym = str(out.at[idx, "Ticker"])
            try:
                out.at[idx, "Sentiment"] = round(sentiment_score(fetch_headlines(sym)), 3)
            except Exception as exc:
                log(f"[warn] {sym}: sentiment failed ({exc})")
            try:
                days = fetch_next_earnings_days(sym)
                if days is not None:
                    out.at[idx, "DaysToEarnings"] = float(days)
            except Exception as exc:
                log(f"[warn] {sym}: earnings lookup failed ({exc})")

    out["Rating"] = rating_from_features(out)
    out["Flags"] = build_flags(out, min_dollar_vol)

    out["MACDhist"] = (out["MACD"] - out["MACDsig"]).round(3)
    out["AvgDollarVol_M"] = (out["AvgDollarVol"] / 1e6).round(1)
    for col, nd in (("Close", 2), ("RSI14", 2), ("SMA50", 2), ("SMA200", 2),
                    ("RS63", 1), ("PctFromHigh", 1), ("ATRpct", 2)):
        out[col] = out[col].round(nd)

    cols = ["Ticker", "Rating", "Score", "Close", "RSI14", "SMA50", "SMA200", "MACDhist",
            "RS63", "PctFromHigh", "ATRpct", "AvgDollarVol_M", "DaysToEarnings",
            "Sentiment", "Flags", "Pattern", "HammerConfirm", "Above SMA50", "RSI<40"]
    out = out[cols].sort_values(by=["Rating", "Score"], ascending=[False, False],
                                kind="mergesort").reset_index(drop=True)
    out.index = out.index + 1
    out.index.name = "Rank"
    return out


# --------------------------------------------------------------------------- #
# Compare with the previous run
# --------------------------------------------------------------------------- #
def attach_previous_ratings(out: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    """Add PrevRating/Change vs the last time each ticker was screened, then save this run."""
    path = cache_dir / "last_run.csv"
    prev = pd.DataFrame(columns=["Ticker", "Rating", "RunDate"])
    if path.exists():
        try:
            prev = pd.read_csv(path)[["Ticker", "Rating", "RunDate"]]
        except Exception:
            prev = pd.DataFrame(columns=["Ticker", "Rating", "RunDate"])

    prev_map = dict(zip(prev["Ticker"], prev["Rating"]))
    out = out.copy()
    out.insert(2, "PrevRating", out["Ticker"].map(prev_map))
    out.insert(3, "Change", (out["Rating"] - out["PrevRating"]).round(1))

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        now = pd.DataFrame({"Ticker": out["Ticker"].values, "Rating": out["Rating"].values,
                            "RunDate": today_et().isoformat()})
        keep = prev[~prev["Ticker"].isin(now["Ticker"])]
        pd.concat([keep, now], ignore_index=True).to_csv(path, index=False)
    except Exception as exc:
        log(f"[warn] could not save run history ({exc})")
    return out


def print_movers(out: pd.DataFrame, n: int = 5) -> None:
    if "Change" not in out.columns:
        return
    ch = out.dropna(subset=["Change"])
    ch = ch[ch["Change"] != 0]
    if ch.empty:
        return
    cols = ["Ticker", "PrevRating", "Rating", "Change"]
    up = ch.sort_values("Change", ascending=False).head(n)
    down = ch.sort_values("Change", ascending=True).head(n)
    print("\nBiggest rating gains since last run:")
    print(up[cols][up["Change"] > 0].to_string(index=False) or "  none")
    print("\nBiggest rating drops since last run:")
    print(down[cols][down["Change"] < 0].to_string(index=False) or "  none")


# --------------------------------------------------------------------------- #
# HTML report (self-contained, sortable, filterable)
# --------------------------------------------------------------------------- #
_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#ddd;--head:#f3f3f3;--acc:#2a6fdb}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--mut:#999;--line:#333;--head:#222}}
body{font:14px system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:16px}
h1{font-size:18px;margin:0 0 4px}p{color:var(--mut);margin:4px 0 12px}
input{padding:6px 8px;margin:0 8px 10px 0;width:220px;background:var(--bg);color:var(--fg);
border:1px solid var(--line);border-radius:6px}
.tabs button{padding:6px 12px;margin:0 6px 10px 0;border:1px solid var(--line);border-radius:6px;
background:var(--bg);color:var(--fg);cursor:pointer}
.tabs button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{background:var(--head);cursor:pointer;position:sticky;top:0}
td:first-child,th:first-child,td.txt,th.txt{text-align:left}
</style></head><body>
<h1>__TITLE__</h1>
<p>__SUBTITLE__</p>
<input id="q" placeholder="Filter by ticker or flag..." oninput="flt()">
<span class="tabs">__TABS__</span>
__SECTIONS__
<script>
function show(i){
document.querySelectorAll('.sec').forEach(function(s,j){s.style.display=(i===j)?'':'none';});
document.querySelectorAll('.tabs button').forEach(function(b,j){b.className=(i===j)?'on':'';});}
function flt(){var q=document.getElementById('q').value.toLowerCase();
document.querySelectorAll('table.tbl tbody tr').forEach(function(r){
r.style.display=r.textContent.toLowerCase().indexOf(q)>-1?'':'none';});}
document.querySelectorAll('table.tbl').forEach(function(tbl){
tbl.querySelectorAll('th').forEach(function(th,i){th.addEventListener('click',function(){
var tb=tbl.tBodies[0];var rows=Array.prototype.slice.call(tb.rows);
var asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
rows.sort(function(a,b){var x=a.cells[i].dataset.v,y=b.cells[i].dataset.v;
var nx=parseFloat(x),ny=parseFloat(y);
if(!isNaN(nx)&&!isNaN(ny))return asc?nx-ny:ny-nx;
return asc?String(x).localeCompare(String(y)):String(y).localeCompare(String(x));});
rows.forEach(function(r){tb.appendChild(r);});});});});
show(0);
</script></body></html>
"""

_HTML_TEXT_COLS = {"Ticker", "Flags", "Pattern", "Signal", "HammerConfirm"}


def _table_html(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "".join(
        f'<th class="{"txt" if c in _HTML_TEXT_COLS else ""}">{html.escape(str(c))}</th>'
        for c in cols
    )
    body_rows: list[str] = []
    for _, row in df.iterrows():
        cells: list[str] = []
        for c in cols:
            v = row[c]
            shown = "" if (not isinstance(v, (str, bool)) and pd.isna(v)) else str(v)
            style = ""
            if c == "Rating" and shown:
                hue = int(round(120 * (float(v) - 1.0) / 9.0))
                style = f' style="background:hsl({hue},55%,45%);color:#fff;font-weight:600"'
            cls = ' class="txt"' if c in _HTML_TEXT_COLS else ""
            cells.append(f'<td{cls}{style} data-v="{html.escape(shown)}">{html.escape(shown)}</td>')
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return ('<div class="wrap"><table class="tbl"><thead><tr>' + head + "</tr></thead><tbody>"
            + "".join(body_rows) + "</tbody></table></div>")


def write_html_report(out: pd.DataFrame, path: str,
                      turnaround: Optional[pd.DataFrame] = None) -> str:
    """Write a self-contained sortable HTML report (tabs: all results, turnaround)."""
    sections = [(f"All results ({len(out)})", out)]
    if turnaround is not None and not turnaround.empty:
        sections.append((f"Turnaround ({len(turnaround)})", turnaround))

    tabs = "".join(f'<button onclick="show({i})">{html.escape(name)}</button>'
                   for i, (name, _) in enumerate(sections))
    blocks = "".join(f'<div class="sec">{_table_html(df)}</div>' for _, df in sections)
    subtitle = (f"Generated {today_et().isoformat()} - rating 1 (bearish setup) to 10 (bullish "
                "setup). Screening heuristics only: not a forecast and not financial advice.")
    page = (_HTML_TEMPLATE.replace("__TITLE__", "Stock Screener Report")
            .replace("__SUBTITLE__", html.escape(subtitle))
            .replace("__TABS__", tabs).replace("__SECTIONS__", blocks))
    try:
        Path(path).write_text(page, encoding="utf-8")
        return f"Saved HTML report to {Path(path).resolve()}"
    except OSError as exc:
        return f"Could not write HTML report ({exc})"


# --------------------------------------------------------------------------- #
# Backtest: did higher ratings historically precede higher returns?
# --------------------------------------------------------------------------- #
def _fmt(x: Any, spec: str = "+.3f") -> str:
    v = _f(x)
    return "n/a" if not np.isfinite(v) else format(v, spec)


def _cs_ic(df: pd.DataFrame, col: str, ret_col: str = "FwdNet",
           min_names: int = 20) -> tuple[float, float, int]:
    """Mean cross-sectional rank correlation (per date) of `col` with forward returns.

    Ranking within each date removes market-wide moves, so it measures stock picking only.
    Returns (mean IC, t-stat, number of dates used).
    """
    vals: list[float] = []
    for _, g in df.groupby("Date"):
        if len(g) < min_names:
            continue
        a, b = g[col].rank(), g[ret_col].rank()
        if a.nunique() < 2 or b.nunique() < 2:
            continue
        c = _f(a.corr(b))
        if np.isfinite(c):
            vals.append(c)
    if len(vals) < 5:
        return float("nan"), float("nan"), len(vals)
    arr = np.asarray(vals, dtype=float)
    sd = float(arr.std(ddof=1))
    t = float(arr.mean() / sd * np.sqrt(arr.size)) if sd > 0 else float("nan")
    return float(arr.mean()), t, int(arr.size)


def quintile_table(df: pd.DataFrame, rating_col: str = "Rating") -> tuple[pd.DataFrame, float]:
    """Net forward return by rating group (lowest first) and the top-minus-bottom spread."""
    if df.empty:
        return pd.DataFrame(), float("nan")
    try:
        groups = pd.qcut(df[rating_col], 5, duplicates="drop", precision=1)
    except ValueError:
        return pd.DataFrame(), float("nan")
    t = (df.assign(Group=groups).groupby("Group", observed=True)
         .agg(Samples=("FwdNet", "size"), AvgRating=(rating_col, "mean"),
              AvgNetPct=("FwdNet", "mean"), AvgExcessSPYPct=("Excess", "mean"),
              HitRatePct=("FwdNet", lambda x: float((x > 0).mean() * 100.0)))
         .reset_index())
    t["Group"] = t["Group"].astype(str)
    spread = float(t["AvgNetPct"].iloc[-1] - t["AvgNetPct"].iloc[0]) if len(t) >= 2 else float("nan")
    return t.round(2), spread


def _composite_eval(df: pd.DataFrame, comp_names: list[str],
                    weights: dict[str, float]) -> tuple[float, float]:
    if df.empty:
        return float("nan"), float("nan")
    tmp = df[["Date", "FwdNet", "Excess"]].copy()
    tmp["Rating"] = combine_components({k: df[k] for k in comp_names}, weights)
    ic = _cs_ic(tmp, "Rating")[0]
    _, spread = quintile_table(tmp)
    return ic, spread


def _simulate_trade(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, s: int,
                    horizon: int, stop_atr: float, atr_pct: float) -> Optional[tuple[float, bool]]:
    """Signal on bar s -> buy next open; stop = stop_atr x ATR below entry; else sell the close
    `horizon` bars after the signal. Gaps through the stop fill at the open. Returns
    (return %, stopped_out) or None if prices are unusable."""
    e, last = s + 1, s + horizon
    if last >= len(c):
        return None
    entry = o[e]
    if not np.isfinite(entry) or entry <= 0:
        return None
    stop = entry * (1.0 - stop_atr * atr_pct / 100.0) if stop_atr > 0 else 0.0
    if stop > 0:
        hit = np.flatnonzero(l[e:last + 1] <= stop)
        if hit.size:
            d = int(hit[0])
            px = o[e + d] if o[e + d] <= stop else stop
            return float((px / entry - 1.0) * 100.0), True
    px = c[last]
    if not np.isfinite(px):
        return None
    return float((px / entry - 1.0) * 100.0), False


def _ticker_events(ticker: str, df: pd.DataFrame, feat: pd.DataFrame,
                   spy_a: Optional[pd.DataFrame], sample_mask: np.ndarray,
                   horizon: int, stop_atr: float, cost_pct: float) -> list[dict[str, Any]]:
    """Simulated trades for baseline days, hammer days and confirmed-hammer days."""
    o, h, l, c = (df[x].to_numpy(np.float64) for x in OHLC_COLS)
    n = len(df)
    dates = df.index
    rsi = feat["RSI14"].to_numpy(np.float64)
    pfh = feat["PctFromHigh"].to_numpy(np.float64)
    atr = feat["ATRpct"].to_numpy(np.float64)
    hammer = candlestick_flags(df)["Hammer"].to_numpy(bool)
    spy_o = spy_a["Open"].to_numpy(np.float64) if spy_a is not None else None
    spy_c = spy_a["Close"].to_numpy(np.float64) if spy_a is not None else None
    events: list[dict[str, Any]] = []

    def add(kind: str, s: int) -> None:
        if s < 60 or s + horizon >= n:
            return
        if not (np.isfinite(rsi[s]) and np.isfinite(pfh[s]) and np.isfinite(atr[s])):
            return
        res = _simulate_trade(o, h, l, c, s, horizon, stop_atr, float(atr[s]))
        if res is None:
            return
        ret, stopped = res
        net = ret - cost_pct
        spy_ret = float("nan")
        if spy_o is not None and spy_c is not None:
            so, sc = spy_o[s + 1], spy_c[s + horizon]
            if np.isfinite(so) and so > 0 and np.isfinite(sc):
                spy_ret = (sc / so - 1.0) * 100.0
        events.append({
            "Ticker": ticker, "Date": dates[s], "Kind": kind,
            "Score": float((100.0 - rsi[s]) + abs(pfh[s]) + atr[s]),
            "Net": net, "Stopped": stopped, "Excess": net - spy_ret,
        })

    for s in np.flatnonzero(sample_mask):
        add("base", int(s))
    for s in np.flatnonzero(hammer):
        add("hammer", int(s))
    conf = np.zeros(n, dtype=bool)
    conf[1:] = hammer[:-1] & (c[1:] > h[:-1])  # day after a hammer closes above its high
    for s in np.flatnonzero(conf):
        add("confirmed", int(s))
    return events


def _summarize_trades(ev: pd.DataFrame, label: str, cut: pd.Timestamp) -> dict[str, Any]:
    late = ev[ev["Date"] > cut]
    return {
        "Strategy": label,
        "Trades": int(len(ev)),
        "AvgNetPct": ev["Net"].mean(),
        "MedianNetPct": ev["Net"].median(),
        "HitRatePct": float((ev["Net"] > 0).mean() * 100.0),
        "AvgExcessVsSPYPct": ev["Excess"].mean(),
        "StoppedPct": float(ev["Stopped"].mean() * 100.0),
        "LateTrades": int(len(late)),
        "LateAvgNetPct": late["Net"].mean() if len(late) else float("nan"),
    }


def run_backtest(
    tickers: list[str],
    period: str,
    horizon: int,
    step: int,
    max_tickers: int,
    chunk_size: int,
    cost_bps: float,
    stop_atr: float,
) -> dict[str, Any]:
    """Backtest the rating (train/test split, per-component IC, suggested weights) and the
    turnaround strategy (confirmation, ATR stops, costs, SPY benchmark).

    Technical inputs only: historical news and earnings aren't available. Returns {} if no data.
    """
    cost_pct = cost_bps / 100.0
    if len(tickers) > max_tickers:
        rng = np.random.RandomState(0)
        tickers = sorted(rng.choice(tickers, size=max_tickers, replace=False).tolist())
        log(f"[info] sampling {max_tickers} tickers for the backtest (use --bt-max-tickers)")

    spy_frames = download_chunk([BENCHMARK], period=period, retries=2)
    spy_df = spy_frames.get(BENCHMARK)
    spy_close = spy_df["Close"] if spy_df is not None else None
    if spy_df is None:
        log(f"[warn] {BENCHMARK} unavailable; excess-vs-SPY and relative strength are neutral")
    sample_dates = spy_df.index[SMA_SLOW::step] if spy_df is not None else None

    rating_rows: list[pd.DataFrame] = []
    events: list[dict[str, Any]] = []
    total = len(tickers)
    min_rows = SMA_SLOW + horizon + 20
    for start in range(0, total, chunk_size):
        chunk = tickers[start:start + chunk_size]
        log(f"[info] backtest data {start + 1}-{start + len(chunk)} of {total} ...")
        frames = download_chunk(chunk, period=period)
        for t, df in frames.items():
            if len(df) < min_rows:
                continue
            try:
                feat = compute_features(df, spy_close)
                parts = rating_components(feat, use_sentiment=False)
                close = feat["Close"].astype(np.float64)
                fwd = (close.shift(-horizon) / close - 1.0) * 100.0 - cost_pct

                spy_a = (spy_df[["Open", "Close"]].reindex(df.index, method="ffill")
                         if spy_df is not None else None)
                if spy_a is not None:
                    sc = spy_a["Close"].astype(np.float64)
                    spy_fwd = (sc.shift(-horizon) / sc - 1.0) * 100.0
                else:
                    spy_fwd = pd.Series(np.nan, index=df.index)

                if sample_dates is not None:
                    sample_mask = np.asarray(df.index.isin(sample_dates), dtype=bool)
                else:
                    sample_mask = np.zeros(len(df), dtype=bool)
                    sample_mask[SMA_SLOW::step] = True

                keep = (sample_mask & feat["SMA200"].notna().to_numpy()
                        & fwd.notna().to_numpy())
                if keep.any():
                    s_df = pd.DataFrame({
                        "Ticker": t, "Date": df.index, "FwdNet": fwd.to_numpy(),
                        "SpyFwd": spy_fwd.to_numpy(),
                        **{k: v.to_numpy() for k, v in parts.items()},
                    })
                    rating_rows.append(s_df[keep])
                events += _ticker_events(t, df, feat, spy_a, sample_mask, horizon,
                                         stop_atr, cost_pct)
            except Exception as exc:
                log(f"[warn] {t}: backtest skipped ({exc})")
        del frames
        if start + chunk_size < total:
            time.sleep(1.0)

    if not rating_rows:
        return {}

    allrows = pd.concat(rating_rows, ignore_index=True)
    allrows["Excess"] = allrows["FwdNet"] - allrows["SpyFwd"]
    comp_names = [k for k in RATING_WEIGHTS if k in allrows.columns]
    allrows["Rating"] = combine_components({k: allrows[k] for k in comp_names}, ACTIVE_WEIGHTS)

    # Time split: first 60% of dates = train, last 40% = test (never peek at the test period).
    dates = np.sort(allrows["Date"].unique())
    cut = pd.Timestamp(dates[min(max(int(len(dates) * 0.6) - 1, 0), len(dates) - 1)])
    train = allrows[allrows["Date"] <= cut]
    test = allrows[allrows["Date"] > cut]

    rating_train, spread_train = quintile_table(train)
    rating_test, spread_test = quintile_table(test)

    # Per-component information coefficient, then suggested weights from TRAIN data only.
    comp_rows: list[dict[str, Any]] = []
    ic_train: dict[str, float] = {}
    for k in comp_names:
        ic_tr, _, _ = _cs_ic(train, k)
        ic_te, t_te, _ = _cs_ic(test, k)
        ic_train[k] = ic_tr
        comp_rows.append({"Component": k, "TrainIC": ic_tr, "TestIC": ic_te,
                          "TestTStat": t_te, "CurrentWeight": ACTIVE_WEIGHTS.get(k, 0.0)})

    sig = {k: (v if np.isfinite(v) and abs(v) >= 0.005 else 0.0) for k, v in ic_train.items()}
    denom = sum(abs(v) for v in sig.values())
    if denom > 0:
        suggested = {k: round(0.9 * sig[k] / denom, 4) for k in comp_names}
        suggested["sentiment"] = 0.10  # can't be backtested (no historical news): keep it small
    else:
        suggested = dict(ACTIVE_WEIGHTS)
    for r in comp_rows:
        r["SuggestedWeight"] = suggested.get(r["Component"], 0.0)

    eval_rows: list[dict[str, Any]] = []
    for label, w in (("Current weights", dict(ACTIVE_WEIGHTS)), ("Suggested weights", suggested)):
        ic_a, sp_a = _composite_eval(train, comp_names, w)
        ic_b, sp_b = _composite_eval(test, comp_names, w)
        eval_rows.append({"Weights": label, "TrainIC": ic_a, "TrainSpreadPct": sp_a,
                          "TestIC": ic_b, "TestSpreadPct": sp_b})

    # Turnaround strategy.
    turn_rows: list[dict[str, Any]] = []
    ev = pd.DataFrame(events)
    if not ev.empty:
        base = ev[ev["Kind"] == "base"]
        hm = ev[ev["Kind"] == "hammer"]
        cf = ev[ev["Kind"] == "confirmed"]
        ref = hm.loc[hm["Date"] <= cut, "Score"]
        if len(ref) < 30:
            ref = hm["Score"]
        hi = float(ref.quantile(2 / 3)) if len(ref) >= 3 else float("nan")
        lo = float(ref.quantile(1 / 3)) if len(ref) >= 3 else float("nan")
        specs = [
            ("Baseline: any stock, sampled days", base),
            ("Hammer", hm),
            ("Hammer + next-day confirmation", cf),
            ("Hammer, top-third Turnaround Score", hm[hm["Score"] >= hi]),
            ("Hammer, bottom-third Turnaround Score", hm[hm["Score"] <= lo]),
            ("Confirmed hammer, top-third score", cf[cf["Score"] >= hi]),
        ]
        for label, sub in specs:
            if len(sub):
                turn_rows.append(_summarize_trades(sub, label, cut))

    return {
        "meta": {
            "samples": len(allrows), "tickers": int(allrows["Ticker"].nunique()),
            "cut": cut, "horizon": horizon, "cost_bps": cost_bps, "stop_atr": stop_atr,
            "spy_ok": spy_df is not None,
            "overall_avg_net": float(allrows["FwdNet"].mean()),
            "overall_avg_excess": float(allrows["Excess"].mean()),
            "weights_changed": denom > 0,
        },
        "rating_train": rating_train, "rating_test": rating_test,
        "spread_train": spread_train, "spread_test": spread_test,
        "components": pd.DataFrame(comp_rows), "evaluation": pd.DataFrame(eval_rows),
        "suggested": suggested, "turnaround": pd.DataFrame(turn_rows),
    }


def print_backtest(res: dict[str, Any]) -> None:
    m = res["meta"]
    print(f"\nBacktest setup: {m['samples']} samples, {m['tickers']} tickers, "
          f"{m['horizon']}-trading-day horizon, {m['cost_bps']:.0f} bps round-trip cost.")
    print(f"Train = dates up to {m['cut'].date()}, Test = later dates (never used for tuning).")
    print(f"Average net return per sample: {_fmt(m['overall_avg_net'], '.2f')}% | "
          f"vs SPY: {_fmt(m['overall_avg_excess'], '+.2f')}%")

    for name, key, sp in (("TRAIN", "rating_train", "spread_train"),
                          ("TEST", "rating_test", "spread_test")):
        tbl = res[key]
        print(f"\nRating groups, {name} period (lowest rating group first):")
        print(tbl.to_string(index=False) if not tbl.empty else "  not enough data")
        print(f"Top minus bottom group: {_fmt(res[sp], '+.2f')} percentage points")

    print("\nWhich ingredients actually predicted returns (cross-sectional rank IC; "
          "positive = bullish signal worked, negative = it worked in reverse):")
    comp = res["components"].copy()
    for c in ("TrainIC", "TestIC", "TestTStat"):
        comp[c] = comp[c].map(lambda v: _fmt(v, "+.3f" if c != "TestTStat" else "+.1f"))
    print(comp.to_string(index=False))

    print("\nCurrent vs suggested weights (suggestion is fit on TRAIN data only):")
    ev = res["evaluation"].copy()
    for c in ("TrainIC", "TestIC"):
        ev[c] = ev[c].map(lambda v: _fmt(v, "+.3f"))
    for c in ("TrainSpreadPct", "TestSpreadPct"):
        ev[c] = ev[c].map(lambda v: _fmt(v, "+.2f"))
    print(ev.to_string(index=False))
    cur, sug = res["evaluation"].iloc[0], res["evaluation"].iloc[1]
    if not m["weights_changed"]:
        print("Read-out: no ingredient showed a usable signal in the train period, so the "
              "suggestion is just your current weights.")
    elif (_f(sug["TestIC"]) >= 0.02 and _f(sug["TestSpreadPct"]) >= 0.3
          and _f(sug["TestIC"]) > _f(cur["TestIC"])
          and _f(sug["TestSpreadPct"]) > _f(cur["TestSpreadPct"])):
        print("Read-out: the suggested weights beat the current ones on the unseen TEST period "
              "by a meaningful margin (IC >= 0.02 and spread >= 0.3 points).")
        print("Try them with --weights-file (saved as suggested_weights.json).")
    else:
        print("Read-out: the suggested weights did NOT clearly beat the current ones on the "
              "unseen test period (needs test IC >= 0.02 and spread >= 0.3 points), so treat "
              "them as unreliable and keep your current weights.")

    print(f"\nTurnaround strategy: buy next open after the signal, {m['stop_atr']:g} x ATR stop, "
          f"sell after {m['horizon']} days, net of costs:")
    turn = res["turnaround"]
    if turn.empty:
        print("  no turnaround trades found")
    else:
        t = turn.copy()
        for c in ("AvgNetPct", "MedianNetPct", "HitRatePct", "AvgExcessVsSPYPct",
                  "StoppedPct", "LateAvgNetPct"):
            t[c] = t[c].map(lambda v: _fmt(v, ".2f"))
        print(t.to_string(index=False))
        base = turn[turn["Strategy"].str.startswith("Baseline")]
        top = turn[turn["Strategy"] == "Hammer, top-third Turnaround Score"]
        if not base.empty and not top.empty:
            b, h = base.iloc[0], top.iloc[0]
            if h["Trades"] >= 100 and _f(h["AvgNetPct"]) > _f(b["AvgNetPct"]) + 0.3 \
                    and _f(h["LateAvgNetPct"]) > _f(b["LateAvgNetPct"]):
                print("Read-out: top-third hammer setups beat the baseline overall and in the "
                      "later dates, in this sample.")
            else:
                print("Read-out: no clear advantage over the baseline (or too few trades).")
    print("\nCaveats: today's index members (survivorship bias), overlapping holding windows, "
          "simplified costs/fills, technical inputs only. Past results do not predict future "
          "returns; this is not financial advice.")


def save_backtest(res: dict[str, Any], out_dir: str, stamp: str) -> list[Path]:
    import json

    saved: list[Path] = []

    def target(name: str) -> Path:
        return resolve_output(name, name, out_dir)

    try:
        both = pd.concat([res["rating_train"].assign(Period="train"),
                          res["rating_test"].assign(Period="test")], ignore_index=True)
        p = target(f"backtest_rating_{stamp}.csv"); both.to_csv(p, index=False); saved.append(p)
        p = target(f"backtest_components_{stamp}.csv")
        res["components"].to_csv(p, index=False); saved.append(p)
        p = target(f"backtest_turnaround_{stamp}.csv")
        res["turnaround"].to_csv(p, index=False); saved.append(p)
        p = target("suggested_weights.json")
        p.write_text(json.dumps(res["suggested"], indent=2), encoding="utf-8"); saved.append(p)
    except OSError as exc:
        log(f"[warn] could not save some backtest files ({exc})")
    return saved


# --------------------------------------------------------------------------- #
# Turnaround ranking: Hammer-pattern mean-reversion candidates
# --------------------------------------------------------------------------- #
TURNAROUND_DROP = ["Rating", "PrevRating", "Change", "Score"]
TURNAROUND_NUMERIC = ["Close", "RSI14", "PctFromHigh", "ATRpct"]
TURNAROUND_COLS = ["Ticker", "Close", "RSI14", "PctFromHigh", "ATRpct", "Pattern", "Signal",
                   "Turnaround Score", "Stop", "Shares", "Position$"]


def turnaround_rank(df: pd.DataFrame, stop_atr: float = DEFAULT_STOP_ATR,
                    risk_dollars: float = DEFAULT_RISK_DOLLARS) -> pd.DataFrame:
    """Hammer-pattern stocks ranked by Turnaround Score, highest first.

    Turnaround_Score = (100 - RSI14) + abs(PctFromHigh) + ATRpct
    Ignores the 1-10 rating entirely (those columns are dropped first).

    Signal: "Pending" = hammer on the latest bar (unconfirmed); "Confirmed" = the previous
    bar was a hammer and the latest bar closed above its high (needs a HammerConfirm column,
    which older CSVs lack, so every row there is "Pending").
    Stop = Close - stop_atr x ATR; Shares = whole shares so a stop-out loses ~risk_dollars.
    """
    df = df.drop(columns=TURNAROUND_DROP, errors="ignore")
    missing = [c for c in ["Ticker", "Pattern", *TURNAROUND_NUMERIC] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.copy()
    for c in TURNAROUND_NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    has_hammer = df["Pattern"].astype(str).str.contains("Hammer", case=False, regex=False, na=False)
    if "HammerConfirm" in df.columns:
        confirmed = df["HammerConfirm"].astype(str).str.strip().str.lower().eq("confirmed")
    else:
        confirmed = pd.Series(False, index=df.index)

    out = df.loc[has_hammer | confirmed].dropna(subset=TURNAROUND_NUMERIC).copy()
    out["Signal"] = np.where(confirmed.loc[out.index] & ~has_hammer.loc[out.index],
                             "Confirmed", "Pending")
    out["Turnaround Score"] = ((100.0 - out["RSI14"]) + out["PctFromHigh"].abs()
                               + out["ATRpct"]).round(2)

    stop = np.maximum(out["Close"] * (1.0 - stop_atr * out["ATRpct"] / 100.0), 0.0)
    risk_ps = (out["Close"] - stop).where(lambda s: s > 0)
    shares = np.floor(risk_dollars / risk_ps).fillna(0).astype(int)
    out["Stop"] = stop.round(2)
    out["Shares"] = shares
    out["Position$"] = (shares * out["Close"]).round(0)

    return (out.sort_values("Turnaround Score", ascending=False, kind="mergesort")
            .reset_index(drop=True)[TURNAROUND_COLS])


def to_markdown(df: pd.DataFrame) -> str:
    """Tiny markdown-table writer (avoids the 'tabulate' dependency)."""
    fmt = {"Close": "{:.2f}", "RSI14": "{:.2f}", "PctFromHigh": "{:.1f}", "ATRpct": "{:.2f}",
           "Turnaround Score": "{:.2f}", "Stop": "{:.2f}", "Shares": "{:,.0f}",
           "Position$": "{:,.0f}"}
    lines = ["| " + " | ".join(df.columns) + " |",
             "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for _, row in df.iterrows():
        cells = [fmt[c].format(row[c]) if c in fmt else str(row[c]) for c in df.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def report_turnaround(result: pd.DataFrame, csv_name: Optional[str], out_dir: str,
                      stop_atr: float, risk_dollars: float) -> None:
    """Print the turnaround table and save it as CSV in the output folder."""
    print("\nTurnaround candidates (Hammer pattern; higher score = more oversold, further "
          "below its high, more volatile):")
    if result.empty:
        print("No stocks matched the Hammer filter.")
        return
    print(to_markdown(result))
    path = resolve_output(csv_name, f"turnaround_results_{today_et().isoformat()}.csv", out_dir)
    try:
        result.to_csv(path, index=False)
        print(f"Saved {len(result)} turnaround rows to {path.resolve()}")
    except OSError as exc:
        print(f"Could not write turnaround CSV ({exc})")
    print(f"Sizing: Stop = Close - {stop_atr:g} x ATR; Shares are sized so a stop-out loses about "
          f"${risk_dollars:,.0f} (before gaps/slippage).")
    print("Signal: 'Pending' = hammer on the latest bar, not yet confirmed; 'Confirmed' = the next "
          "bar closed above the hammer's high. Screening aid only, not advice.")


def run_turnaround_from_csv(args: argparse.Namespace) -> int:
    """Re-rank an existing screener CSV without any downloads."""
    try:
        df = pd.read_csv(args.from_csv)
        result = turnaround_rank(df, args.stop_atr, args.risk_dollars)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    report_turnaround(result, args.csv, args.out_dir, args.stop_atr, args.risk_dollars)
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lightweight stock screener")
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Ticker symbols as Yahoo Finance writes them (e.g. BRK-B, VOD.L)")
    p.add_argument("--sp500", action="store_true", help="Shortcut for --universe sp500")
    p.add_argument("--universe", nargs="+", choices=sorted(INDEXES), default=[],
                   help="Index lists to screen: " + ", ".join(sorted(INDEXES)))
    p.add_argument("--file", default=None,
                   help="Text/CSV file of tickers (newline, comma or space separated)")
    p.add_argument("--period", default=DEFAULT_PERIOD,
                   help="yfinance lookback period (default: %(default)s)")
    p.add_argument("--csv", default=None,
                   help="Results CSV filename (screening / --from-csv modes; "
                        "default: screener_results_YYYY-MM-DD.csv)")
    p.add_argument("--out-dir", default=OUTPUT_DIR_DEFAULT,
                   help="Folder for CSV/HTML outputs (default: %(default)s)")
    p.add_argument("--html", nargs="?", const="auto", default=None, metavar="PATH",
                   help="Also write a sortable HTML report (default name if no PATH)")
    p.add_argument("--chunk", type=int, default=DEFAULT_CHUNK,
                   help="Tickers per download batch (default: %(default)s)")
    p.add_argument("--top-news", type=int, default=DEFAULT_TOP_DETAIL,
                   help="Fetch news sentiment and earnings dates for the top N; 0 = skip "
                        "(default: %(default)s)")
    p.add_argument("--show", type=int, default=DEFAULT_SHOW,
                   help="Rows to print to the terminal (default: %(default)s)")
    p.add_argument("--min-dollar-vol", type=float, default=DEFAULT_MIN_DOLLAR_VOL,
                   help="Flag LowLiquidity below this 20-day avg dollar volume "
                        "(default: %(default).0f)")
    p.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT,
                   help="Price cache directory (default: %(default)s)")
    p.add_argument("--no-cache", action="store_true",
                   help="Disable the price cache and run-to-run comparison")
    p.add_argument("--refresh", action="store_true",
                   help="Ignore cached prices and re-download everything")
    p.add_argument("--quiet", action="store_true",
                   help="Hide [info] progress lines (handy for scheduled/cron runs)")
    p.add_argument("--weights-file", default=None, metavar="PATH",
                   help="JSON rating weights, e.g. screener_output/suggested_weights.json")
    p.add_argument("--no-turnaround", action="store_true",
                   help="Skip the automatic Hammer-pattern turnaround list after screening")
    p.add_argument("--from-csv", default=None, metavar="PATH",
                   help="Skip screening: re-rank a saved screener CSV by Turnaround Score")
    p.add_argument("--stop-atr", type=float, default=DEFAULT_STOP_ATR,
                   help="Stop distance in ATRs for turnaround sizing and backtests "
                        "(default: %(default)s)")
    p.add_argument("--risk-dollars", type=float, default=DEFAULT_RISK_DOLLARS,
                   help="Dollars risked per turnaround trade for share sizing "
                        "(default: %(default).0f)")
    p.add_argument("--backtest", action="store_true",
                   help="Backtest the rating and the turnaround strategy instead of screening")
    p.add_argument("--bt-period", default="5y", help="Backtest history (default: %(default)s)")
    p.add_argument("--bt-horizon", type=int, default=21,
                   help="Holding period in trading days (default: %(default)s)")
    p.add_argument("--bt-step", type=int, default=5,
                   help="Sample every N trading days (default: %(default)s)")
    p.add_argument("--bt-max-tickers", type=int, default=150,
                   help="Max tickers sampled for the backtest (default: %(default)s)")
    p.add_argument("--bt-cost-bps", type=float, default=DEFAULT_COST_BPS,
                   help="Round-trip trading cost in basis points (default: %(default).0f)")
    return p.parse_args(argv)


def resolve_tickers(args: argparse.Namespace) -> list[str]:
    tickers: list[str] = []
    universes = list(dict.fromkeys(args.universe + (["sp500"] if args.sp500 else [])))
    given_anything = bool(universes or args.file or args.tickers)
    if args.backtest and not given_anything:
        universes = ["sp500"]  # a 4-stock backtest would mean nothing
    for name in universes:
        tickers += get_index_tickers(name)
    if args.file:
        tickers += load_ticker_file(args.file)
    if args.tickers:
        tickers += [_normalize_symbol(t) for t in args.tickers if str(t).strip()]
    if not tickers:
        if given_anything or args.backtest:
            log("[warn] no tickers found from your options; using the default watchlist")
        tickers = list(DEFAULT_WATCHLIST)
    return list(dict.fromkeys(t for t in tickers if t))


def main(argv: Optional[list[str]] = None) -> int:
    global QUIET
    args = parse_args(argv)
    QUIET = bool(args.quiet)

    if args.weights_file:
        try:
            weights = load_weights(args.weights_file)
        except (OSError, ValueError) as exc:  # JSONDecodeError is a ValueError
            print(f"Error reading weights file: {exc}", file=sys.stderr)
            return 1
        ACTIVE_WEIGHTS.clear()
        ACTIVE_WEIGHTS.update(weights)
        log(f"[info] using custom rating weights from {args.weights_file}")

    if args.from_csv:
        return run_turnaround_from_csv(args)

    tickers = resolve_tickers(args)
    if not tickers:
        log("No valid tickers supplied.")
        return 1
    chunk = max(1, int(args.chunk))
    stamp = today_et().isoformat()

    # ---- backtest mode ----
    if args.backtest:
        res = run_backtest(
            tickers, args.bt_period, max(1, args.bt_horizon), max(1, args.bt_step),
            max(1, args.bt_max_tickers), chunk, float(args.bt_cost_bps), float(args.stop_atr),
        )
        if not res:
            print("Backtest produced no results (check network/tickers).")
            return 1
        print_backtest(res)
        for path in save_backtest(res, args.out_dir, stamp):
            print(f"Saved {path.resolve()}")
        print("To try the suggested rating weights: python stock-screener.py --sp500 "
              f"--weights-file {Path(args.out_dir) / 'suggested_weights.json'}")
        return 0

    # ---- screening mode ----
    if len(tickers) > 600:
        log(f"[info] {len(tickers)} tickers: first run can take 10+ minutes on low-power "
            "hardware; later runs are much faster thanks to the cache")
    cache_dir = None if args.no_cache else Path(args.cache_dir)

    ranking = build_ranking(
        tickers, args.period, chunk_size=chunk, top_detail=max(0, int(args.top_news)),
        cache_dir=cache_dir, refresh=bool(args.refresh),
        min_dollar_vol=float(args.min_dollar_vol),
    )
    if ranking.empty:
        print("No results: no ticker returned usable data (check network/tickers).")
        return 1

    if cache_dir is not None:
        ranking = attach_previous_ratings(ranking, cache_dir)

    csv_path = resolve_output(args.csv, f"screener_results_{stamp}.csv", args.out_dir)
    try:
        ranking.to_csv(csv_path)
        saved = f"Saved {len(ranking)} rows to {csv_path.resolve()}"
    except OSError as exc:
        saved = f"Could not write CSV ({exc})"

    display_cols = [c for c in ("Ticker", "Rating", "Change", "Score", "Close", "RSI14", "RS63",
                                "PctFromHigh", "ATRpct", "Sentiment", "Flags", "Pattern")
                    if c in ranking.columns]
    shown = ranking[display_cols].head(max(1, int(args.show)))
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(shown.to_string())
    if len(ranking) > len(shown):
        print(f"... showing top {len(shown)} of {len(ranking)}")
    print_movers(ranking)
    print(saved)

    turnaround: Optional[pd.DataFrame] = None
    if not args.no_turnaround:
        try:
            turnaround = turnaround_rank(ranking.reset_index(drop=True),
                                         float(args.stop_atr), float(args.risk_dollars))
            report_turnaround(turnaround, None, args.out_dir,
                              float(args.stop_atr), float(args.risk_dollars))
        except ValueError as exc:
            print(f"Turnaround skipped: {exc}")

    if args.html is not None:
        html_path = resolve_output(None if args.html == "auto" else args.html,
                                   f"screener_report_{stamp}.html", args.out_dir)
        print(write_html_report(ranking, str(html_path), turnaround))
    return 0


if __name__ == "__main__":
    sys.exit(main())
