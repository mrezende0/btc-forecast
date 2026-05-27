"""Backfill histórico de notícias via GDELT 2.0 Doc API (gratuito, sem chave).

Estratégia: chunks diários — GDELT Doc API retorna no máx 250 artigos por query, sem
paginação. Pra histórico denso (crypto tem MUITAS notícias/dia), chunkar por hora é
seguro mas lento. Default: diário. Pra dias densos, usar `--hourly`.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import polars as pl
import requests

API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Query padrão — cobre BTC + grandes nomes + termos genéricos.
# GDELT aceita OR/AND/aspas. Mantém em inglês (GDELT é multilingue mas indexa em EN).
DEFAULT_QUERY = (
    '("bitcoin" OR "btc" OR "ethereum" OR "ether" OR "cryptocurrency" '
    'OR "crypto" OR "blockchain") sourcelang:english'
)

MAX_RECORDS = 250


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def _fetch_window(
    start: datetime,
    end: datetime,
    query: str = DEFAULT_QUERY,
    retries: int = 3,
) -> list[dict]:
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": MAX_RECORDS,
        "sort": "datedesc",
        "startdatetime": _fmt(start),
        "enddatetime": _fmt(end),
    }
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(API, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            # GDELT pode retornar HTML em erro — JSON nem sempre vem
            ctype = r.headers.get("content-type", "")
            if "json" not in ctype:
                return []
            return r.json().get("articles", []) or []
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    if last_err:
        raise last_err
    return []


def fetch_range(
    start: datetime,
    end: datetime,
    query: str = DEFAULT_QUERY,
    chunk: str = "day",
    sleep_between: float = 1.0,
) -> pl.DataFrame:
    """Itera janelas (day|hour) entre start e end, agrega resultados.

    Args:
        start, end: datetimes UTC. Se naive, assumimos UTC.
        chunk: 'day' (recomendado pra histórico) ou 'hour' (dias densos).
        sleep_between: respeito ao rate limit GDELT (sem doc oficial; 1s é conservador).
    """
    start = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start
    end = end.replace(tzinfo=timezone.utc) if end.tzinfo is None else end
    delta = timedelta(days=1) if chunk == "day" else timedelta(hours=1)

    all_rows: list[dict] = []
    cursor = start
    while cursor < end:
        nxt = min(cursor + delta, end)
        try:
            articles = _fetch_window(cursor, nxt, query)
        except Exception as e:
            print(f"[gdelt]  ⚠️  {cursor.isoformat()}: {e}")
            articles = []

        for a in articles:
            all_rows.append(
                {
                    "url": a.get("url"),
                    "title": a.get("title"),
                    "seendate": a.get("seendate"),  # YYYYMMDDHHMMSS UTC
                    "domain": a.get("domain"),
                    "language": a.get("language"),
                    "sourcecountry": a.get("sourcecountry"),
                    "source": "gdelt",
                }
            )

        cursor = nxt
        time.sleep(sleep_between)

    if not all_rows:
        return pl.DataFrame()

    df = pl.DataFrame(all_rows)
    # Parse seendate em ts unix (segundos UTC)
    df = df.with_columns(
        pl.col("seendate")
        .str.strptime(pl.Datetime("us", time_zone="UTC"), "%Y%m%d%H%M%S", strict=False)
        .alias("published_at")
    )
    df = df.with_columns(
        (pl.col("published_at").dt.timestamp("ms") // 1000).alias("published_ts")
    )
    return (
        df.drop_nulls(subset=["url", "title", "published_ts"])
        .unique(subset=["url"], keep="first")
        .sort("published_ts")
    )
