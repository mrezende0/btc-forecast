"""Fetchers Bybit V5. Alternativa a Binance fapi (geo-bloqueado em runners US).

Endpoints:
  /v5/market/kline             — OHLCV spot ou linear (perp)
  /v5/market/funding/history   — funding rate 8h
  /v5/market/open-interest     — OI histórico (paginado via cursor)
  /v5/market/account-ratio     — long/short ratio (contas Bybit)

REFERÊNCIA: valores diferem de Binance (funding/OI/LS são por-exchange).
Trends e z-scores são altamente correlacionados — features ML continuam válidas.
"""
from __future__ import annotations

import time

import polars as pl
import requests

BASE = "https://api.bybit.com"
KLINE = BASE + "/v5/market/kline"
FUNDING = BASE + "/v5/market/funding/history"
OI = BASE + "/v5/market/open-interest"
LS_RATIO = BASE + "/v5/market/account-ratio"

INTERVAL_MIN = "15"
INTERVAL_MS = 15 * 60 * 1000
KLINE_LIMIT = 1000  # Bybit V5 max
GENERIC_LIMIT = 200
LS_LIMIT = 500
SLEEP = 0.2


def _get(url: str, params: dict, retries: int = 4) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            body = r.json()
            if body.get("retCode") != 0:
                raise RuntimeError(f"bybit retCode={body.get('retCode')} msg={body.get('retMsg')}")
            return body["result"]
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    if last:
        raise last
    return {}


# ----------------------------------------------------------------- klines
def fetch_klines(start_ms: int, end_ms: int | None = None, symbol: str = "BTCUSDT",
                 category: str = "spot") -> pl.DataFrame:
    """OHLCV 15m. category: 'spot' ou 'linear' (USDT-perp).

    Bybit retorna lista em ordem decrescente (mais recente primeiro).
    Paginação: avança end pra trás (start menor que primeiro retornado).
    """
    end_ms = end_ms or int(time.time() * 1000)
    rows: list[list] = []
    cur_end = end_ms
    seen: set[int] = set()
    while cur_end > start_ms:
        result = _get(KLINE, {
            "category": category,
            "symbol": symbol,
            "interval": INTERVAL_MIN,
            "start": start_ms,
            "end": cur_end,
            "limit": KLINE_LIMIT,
        })
        batch = result.get("list", [])
        if not batch:
            break
        # mais recente primeiro → ordenar por timestamp
        oldest_ts = int(batch[-1][0])
        if oldest_ts in seen:
            break
        seen.add(oldest_ts)
        rows.extend(batch)
        if len(batch) < KLINE_LIMIT:
            break
        cur_end = oldest_ts - 1
        time.sleep(SLEEP)

    if not rows:
        return pl.DataFrame()

    by_ts: dict[int, list] = {int(r[0]): r for r in rows}
    parsed = [
        {
            "open_time": ts,
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "volume": float(r[5]),
            "quote_volume": float(r[6]),
        }
        for ts, r in sorted(by_ts.items())
    ]
    return pl.DataFrame(parsed)


def fetch_perp_klines(start_ms: int, end_ms: int | None = None,
                      symbol: str = "BTCUSDT") -> pl.DataFrame:
    """Linear USDT-perp klines. Renomeia colunas pra perp_* (compat com features)."""
    df = fetch_klines(start_ms, end_ms, symbol=symbol, category="linear")
    if df.is_empty():
        return df
    return df.rename({
        "open": "perp_open",
        "high": "perp_high",
        "low": "perp_low",
        "close": "perp_close",
        "volume": "perp_volume",
        "quote_volume": "perp_quote_volume",
    })


# ----------------------------------------------------------------- funding
def fetch_funding(start_ms: int, end_ms: int | None = None,
                  symbol: str = "BTCUSDT") -> pl.DataFrame:
    """Funding rate history (8h cadence)."""
    end_ms = end_ms or int(time.time() * 1000)
    rows: list[dict] = []
    cur_end = end_ms
    seen: set[int] = set()
    while cur_end > start_ms:
        result = _get(FUNDING, {
            "category": "linear",
            "symbol": symbol,
            "startTime": start_ms,
            "endTime": cur_end,
            "limit": GENERIC_LIMIT,
        })
        batch = result.get("list", [])
        if not batch:
            break
        oldest_ts = int(batch[-1]["fundingRateTimestamp"])
        if oldest_ts in seen:
            break
        seen.add(oldest_ts)
        rows.extend(batch)
        if len(batch) < GENERIC_LIMIT:
            break
        cur_end = oldest_ts - 1
        time.sleep(SLEEP)

    if not rows:
        return pl.DataFrame()

    by_ts: dict[int, dict] = {int(r["fundingRateTimestamp"]): r for r in rows}
    return pl.DataFrame([
        {
            "funding_time": ts,
            "funding_rate": float(r["fundingRate"]),
        }
        for ts, r in sorted(by_ts.items())
    ])


# ----------------------------------------------------------------- open interest
def fetch_open_interest(start_ms: int, end_ms: int | None = None,
                        symbol: str = "BTCUSDT") -> pl.DataFrame:
    """OI histórico 15min via cursor pagination."""
    end_ms = end_ms or int(time.time() * 1000)
    rows: list[dict] = []
    cursor: str | None = None
    while True:
        params = {
            "category": "linear",
            "symbol": symbol,
            "intervalTime": "15min",
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": GENERIC_LIMIT,
        }
        if cursor:
            params["cursor"] = cursor
        result = _get(OI, params)
        batch = result.get("list", [])
        if not batch:
            break
        rows.extend(batch)
        cursor = result.get("nextPageCursor")
        if not cursor or len(batch) < GENERIC_LIMIT:
            break
        # safety: bail se timestamp do batch já é menor que start_ms
        oldest_ts = int(batch[-1]["timestamp"])
        if oldest_ts <= start_ms:
            break
        time.sleep(SLEEP)

    if not rows:
        return pl.DataFrame()

    by_ts: dict[int, dict] = {int(r["timestamp"]): r for r in rows}
    return pl.DataFrame([
        {
            "open_time": ts,
            "oi": float(r["openInterest"]),
        }
        for ts, r in sorted(by_ts.items())
    ])


# ----------------------------------------------------------------- long/short ratio
def fetch_long_short_ratio(start_ms: int, end_ms: int | None = None,
                           symbol: str = "BTCUSDT") -> pl.DataFrame:
    """Account-based long/short ratio (Bybit users). period=15min."""
    end_ms = end_ms or int(time.time() * 1000)
    rows: list[dict] = []
    cur_end = end_ms
    seen: set[int] = set()
    while cur_end > start_ms:
        result = _get(LS_RATIO, {
            "category": "linear",
            "symbol": symbol,
            "period": "15min",
            "startTime": start_ms,
            "endTime": cur_end,
            "limit": LS_LIMIT,
        })
        batch = result.get("list", [])
        if not batch:
            break
        oldest_ts = int(batch[-1]["timestamp"])
        if oldest_ts in seen:
            break
        seen.add(oldest_ts)
        rows.extend(batch)
        if len(batch) < LS_LIMIT:
            break
        cur_end = oldest_ts - 1
        time.sleep(SLEEP)

    if not rows:
        return pl.DataFrame()

    by_ts: dict[int, dict] = {int(r["timestamp"]): r for r in rows}
    return pl.DataFrame([
        {
            "open_time": ts,
            "ls_buy_ratio": float(r["buyRatio"]),
            "ls_sell_ratio": float(r["sellRatio"]),
            "ls_ratio": float(r["buyRatio"]) / max(1e-9, float(r["sellRatio"])),
        }
        for ts, r in sorted(by_ts.items())
    ])
