"""Fetchers Binance fapi para derivativos.

LIMITAÇÃO: endpoints fapi `openInterestHist`, `topLongShortAccountRatio`,
`takerlongshortRatio` retêm apenas ~30 dias. Backfill profundo NÃO é possível
no plano free. Use cron incremental pra acumular histórico ao longo do tempo.

Env vars opcionais (override pra CF Worker proxy):
  BINANCE_FAPI_BASE  default https://fapi.binance.com
  PROXY_TOKEN        se set, envia como header X-Proxy-Token
"""
from __future__ import annotations

import os
import time

import polars as pl
import requests

_PROXY = (os.environ.get("PROXY_BASE") or "").rstrip("/")
_FAPI = (os.environ.get("BINANCE_FAPI_BASE")
         or (f"{_PROXY}/binance-fapi" if _PROXY else "https://fapi.binance.com")).rstrip("/")
_TOKEN = os.environ.get("PROXY_TOKEN", "")
_HEADERS = {"X-Proxy-Token": _TOKEN} if _TOKEN else {}

OI_HIST = f"{_FAPI}/futures/data/openInterestHist"
TOP_LS_ACC = f"{_FAPI}/futures/data/topLongShortAccountRatio"
TOP_LS_POS = f"{_FAPI}/futures/data/topLongShortPositionRatio"
GLOBAL_LS_ACC = f"{_FAPI}/futures/data/globalLongShortAccountRatio"
TAKER_RATIO = f"{_FAPI}/futures/data/takerlongshortRatio"

PERIOD = "15m"
PERIOD_MS = 15 * 60 * 1000
LIMIT = 500  # fapi data max
SLEEP = 0.3


def _get(url: str, params: dict, retries: int = 4) -> list:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20, headers=_HEADERS)
            if r.status_code == 451:
                print(f"[deriv] WARN: 451 geo-block em {url}")
                return []
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    if last:
        raise last
    return []


def _paginate(url: str, symbol: str, start_ms: int, end_ms: int | None,
              extra: dict | None = None) -> list[dict]:
    """Pagina BACKWARD: Binance fapi data ignora startTime e retorna os últimos
    LIMIT pontos até endTime. Iteramos avançando endTime pra trás até cobrir
    o range [start_ms, end_ms].
    """
    end_ms = end_ms or int(time.time() * 1000)
    rows: list[dict] = []
    extra = extra or {}
    cur_end = end_ms
    seen_first_ts: set[int] = set()
    while cur_end > start_ms:
        batch = _get(url, {
            "symbol": symbol,
            "period": PERIOD,
            "startTime": start_ms,
            "endTime": cur_end,
            "limit": LIMIT,
            **extra,
        })
        if not batch:
            break
        first_ts = int(batch[0].get("timestamp", batch[0].get("time", 0)))
        if first_ts in seen_first_ts:  # protege contra loop infinito
            break
        seen_first_ts.add(first_ts)
        rows.extend(batch)
        if len(batch) < LIMIT:
            break
        cur_end = first_ts - 1  # próxima janela termina antes do mais antigo já coletado
        time.sleep(SLEEP)
    # dedup por timestamp + ordena ascendente
    by_ts: dict[int, dict] = {}
    for r in rows:
        ts = int(r.get("timestamp", r.get("time", 0)))
        by_ts[ts] = r
    return [by_ts[k] for k in sorted(by_ts)]


def fetch_open_interest_hist(start_ms: int, symbol: str = "BTCUSDT",
                             end_ms: int | None = None) -> pl.DataFrame:
    """Open Interest historical em USDT-perp. Retém ~30 dias."""
    rows = _paginate(OI_HIST, symbol, start_ms, end_ms)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame([
        {
            "open_time": int(r["timestamp"]),
            "oi": float(r["sumOpenInterest"]),
            "oi_value_usd": float(r["sumOpenInterestValue"]),
        }
        for r in rows
    ])


def fetch_top_long_short_account(start_ms: int, symbol: str = "BTCUSDT",
                                 end_ms: int | None = None) -> pl.DataFrame:
    """Top traders long/short ratio por CONTAS."""
    rows = _paginate(TOP_LS_ACC, symbol, start_ms, end_ms)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame([
        {
            "open_time": int(r["timestamp"]),
            "top_ls_acc_long": float(r["longAccount"]),
            "top_ls_acc_short": float(r["shortAccount"]),
            "top_ls_acc_ratio": float(r["longShortRatio"]),
        }
        for r in rows
    ])


def fetch_top_long_short_position(start_ms: int, symbol: str = "BTCUSDT",
                                  end_ms: int | None = None) -> pl.DataFrame:
    """Top traders long/short ratio por TAMANHO DE POSIÇÃO (mais informativo que conta)."""
    rows = _paginate(TOP_LS_POS, symbol, start_ms, end_ms)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame([
        {
            "open_time": int(r["timestamp"]),
            "top_ls_pos_long": float(r["longAccount"]),
            "top_ls_pos_short": float(r["shortAccount"]),
            "top_ls_pos_ratio": float(r["longShortRatio"]),
        }
        for r in rows
    ])


def fetch_global_long_short_account(start_ms: int, symbol: str = "BTCUSDT",
                                    end_ms: int | None = None) -> pl.DataFrame:
    """Long/short ratio de TODAS as contas (retail-heavy)."""
    rows = _paginate(GLOBAL_LS_ACC, symbol, start_ms, end_ms)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame([
        {
            "open_time": int(r["timestamp"]),
            "global_ls_long": float(r["longAccount"]),
            "global_ls_short": float(r["shortAccount"]),
            "global_ls_ratio": float(r["longShortRatio"]),
        }
        for r in rows
    ])


def fetch_taker_ratio(start_ms: int, symbol: str = "BTCUSDT",
                      end_ms: int | None = None) -> pl.DataFrame:
    """Taker buy/sell ratio (agressão direcional via ordens market)."""
    rows = _paginate(TAKER_RATIO, symbol, start_ms, end_ms)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame([
        {
            "open_time": int(r["timestamp"]),
            "taker_buy_vol": float(r["buyVol"]),
            "taker_sell_vol": float(r["sellVol"]),
            "taker_buy_sell_ratio": float(r["buySellRatio"]),
        }
        for r in rows
    ])
