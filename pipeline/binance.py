"""Fetcher Binance: OHLCV 15m + funding rate.

Filtra velas em formação (only close_time <= now - 1min) pra evitar gravar dados
que vão mudar.
"""
from __future__ import annotations

import time

import polars as pl
import requests

SPOT_KLINES = "https://api.binance.com/api/v3/klines"
FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"

SYMBOL = "BTCUSDT"
INTERVAL = "15m"
INTERVAL_MS = 15 * 60 * 1000
CLOSED_BUFFER_MS = 60 * 1000  # 1min de folga após a vela fechar


def _now_ms() -> int:
    return int(time.time() * 1000)


def fetch_klines(start_ms: int, end_ms: int | None = None) -> pl.DataFrame:
    end_ms = end_ms or _now_ms()
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        r = requests.get(
            SPOT_KLINES,
            params={
                "symbol": SYMBOL,
                "interval": INTERVAL,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=20,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + INTERVAL_MS
        time.sleep(0.25)
        if len(batch) < 1000:
            break

    if not rows:
        return pl.DataFrame()

    parsed = [
        {
            "open_time": int(r[0]),
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "volume": float(r[5]),
            "close_time": int(r[6]),
            "quote_volume": float(r[7]),
            "trades": int(r[8]),
        }
        for r in rows
    ]
    df = pl.DataFrame(parsed)
    cutoff = _now_ms() - CLOSED_BUFFER_MS
    return df.filter(pl.col("close_time") <= cutoff)


def fetch_funding(start_ms: int, end_ms: int | None = None) -> pl.DataFrame:
    end_ms = end_ms or _now_ms()
    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        r = requests.get(
            FUNDING,
            params={
                "symbol": SYMBOL,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=20,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1]["fundingTime"] + 1
        time.sleep(0.25)
        if len(batch) < 1000:
            break

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(
        [
            {
                "funding_time": int(r["fundingTime"]),
                "funding_rate": float(r["fundingRate"]),
            }
            for r in rows
        ]
    )
