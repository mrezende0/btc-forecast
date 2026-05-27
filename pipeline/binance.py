"""Fetcher Binance: OHLCV 15m + funding rate.

Filtra velas em formação (only close_time <= now - 1min) pra evitar gravar dados
que vão mudar.
"""
from __future__ import annotations

import time

import polars as pl
import requests

SPOT_KLINES = "https://data-api.binance.vision/api/v3/klines"
FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
PERP_KLINES = "https://fapi.binance.com/fapi/v1/klines"  # perpetual futures

SYMBOL = "BTCUSDT"  # default — pode ser override via parâmetro
INTERVAL = "15m"
INTERVAL_MS = 15 * 60 * 1000
CLOSED_BUFFER_MS = 60 * 1000  # 1min de folga após a vela fechar


def _now_ms() -> int:
    return int(time.time() * 1000)


def _get_with_retries(url: str, params: dict, timeout: int = 30, retries: int = 4) -> requests.Response:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    if last:
        raise last


def fetch_klines(start_ms: int, end_ms: int | None = None, symbol: str = SYMBOL) -> pl.DataFrame:
    end_ms = end_ms or _now_ms()
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        r = _get_with_retries(
            SPOT_KLINES,
            params={
                "symbol": symbol,
                "interval": INTERVAL,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=30,
        )
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

    # Binance kline schema: open_t, O, H, L, C, vol, close_t, quote_vol,
    # trades, taker_buy_vol, taker_buy_quote_vol, ignore
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
            "taker_buy_volume": float(r[9]) if len(r) > 9 else 0.0,
            "taker_buy_quote_volume": float(r[10]) if len(r) > 10 else 0.0,
        }
        for r in rows
    ]
    df = pl.DataFrame(parsed)
    cutoff = _now_ms() - CLOSED_BUFFER_MS
    return df.filter(pl.col("close_time") <= cutoff)


def fetch_funding(start_ms: int, end_ms: int | None = None, symbol: str = SYMBOL) -> pl.DataFrame:
    end_ms = end_ms or _now_ms()
    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        r = requests.get(
            FUNDING,
            params={
                "symbol": symbol,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=20,
        )
        if r.status_code == 451:
            # fapi sem mirror público — runner geo-bloqueado. Skip funding.
            print("[funding] WARN: 451 geo-block, pulando coleta")
            return pl.DataFrame()
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


def fetch_perp_klines(start_ms: int, end_ms: int | None = None, symbol: str = SYMBOL) -> pl.DataFrame:
    """OHLCV de perpetuals (fapi) — usado pra computar basis vs spot.

    Mesmo schema do spot pra simplificar joins por open_time.
    """
    end_ms = end_ms or _now_ms()
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        r = requests.get(
            PERP_KLINES,
            params={
                "symbol": symbol,
                "interval": INTERVAL,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1500,  # fapi max é 1500
            },
            timeout=20,
        )
        if r.status_code == 451:
            print("[perp] WARN: 451 geo-block, pulando coleta perp")
            return pl.DataFrame()
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + INTERVAL_MS
        time.sleep(0.25)
        if len(batch) < 1500:
            break

    if not rows:
        return pl.DataFrame()

    parsed = [
        {
            "open_time": int(r[0]),
            "perp_open": float(r[1]),
            "perp_high": float(r[2]),
            "perp_low": float(r[3]),
            "perp_close": float(r[4]),
            "perp_volume": float(r[5]),
            "perp_close_time": int(r[6]),
            "perp_quote_volume": float(r[7]),
            "perp_trades": int(r[8]),
            "perp_taker_buy_volume": float(r[9]) if len(r) > 9 else 0.0,
            "perp_taker_buy_quote_volume": float(r[10]) if len(r) > 10 else 0.0,
        }
        for r in rows
    ]
    df = pl.DataFrame(parsed)
    cutoff = _now_ms() - CLOSED_BUFFER_MS
    return df.filter(pl.col("perp_close_time") <= cutoff)
