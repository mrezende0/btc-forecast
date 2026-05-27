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

HEADERS = {
    "User-Agent": "btc-forecast/0.1 (research; matheus.rezende@labrynth.ai)",
}

# GDELT pede "1 req a cada 5s" mas na prática global rate é mais agressivo.
MIN_SLEEP = 7.0

# Query simples — GDELT rejeita OR complexos + sourcelang com "phrase too short".
# "bitcoin" puro retorna ~250 manchetes/dia, cobertura excelente pra sentiment BTC.
# Filtramos language=English no client side (campo `language` no response).
DEFAULT_QUERY = "bitcoin"

MAX_RECORDS = 250


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def _fetch_window(
    start: datetime,
    end: datetime,
    query: str = DEFAULT_QUERY,
    retries: int = 2,
    timeout: int = 30,
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
            r = requests.get(API, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 429:
                time.sleep(MIN_SLEEP * (attempt + 1))
                continue
            r.raise_for_status()
            # GDELT pode retornar texto puro em rate-limit/erro
            ctype = r.headers.get("content-type", "")
            body = r.text
            if "limit requests" in body.lower():
                time.sleep(MIN_SLEEP * (attempt + 1))
                continue
            if "json" not in ctype:
                return []
            # GDELT às vezes retorna JSON malformado (escapes inválidos em URLs)
            try:
                return r.json().get("articles", []) or []
            except ValueError:
                # tenta strict=False
                import json
                try:
                    return json.loads(body, strict=False).get("articles", []) or []
                except Exception:
                    return []  # janela perdida, segue
        except Exception as e:
            last_err = e
            time.sleep(MIN_SLEEP * (attempt + 1))
    if last_err:
        raise last_err
    return []


def _finalize(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows)
    # GDELT 2.0 seendate vem como "20241201T120000Z"
    df = df.with_columns(
        pl.col("seendate")
        .str.strptime(pl.Datetime("us", time_zone="UTC"), "%Y%m%dT%H%M%SZ", strict=False)
        .alias("published_at")
    )
    df = df.with_columns(
        (pl.col("published_at").dt.timestamp("ms") // 1000).alias("published_ts")
    )
    return (
        df.drop_nulls(subset=["url", "title", "published_ts"])
        .filter(pl.col("language") == "English")
        .unique(subset=["url"], keep="first")
        .sort("published_ts")
    )


def fetch_range(
    start: datetime,
    end: datetime,
    query: str = DEFAULT_QUERY,
    chunk: str = "day",
    sleep_between: float = MIN_SLEEP,
    progress_every: int = 30,
    flush_callback=None,
) -> pl.DataFrame:
    """Itera janelas (day|hour) entre start e end, agrega resultados.

    Args:
        start, end: datetimes UTC. Se naive, assumimos UTC.
        chunk: 'day' (recomendado pra histórico) ou 'hour' (dias densos).
        sleep_between: respeito ao rate limit GDELT (>=5s).
        progress_every: imprime status a cada N janelas.
        flush_callback: opcional, função(df) chamada a cada `progress_every` janelas
            com os novos resultados acumulados — permite persistência incremental.
    """
    start = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start
    end = end.replace(tzinfo=timezone.utc) if end.tzinfo is None else end
    delta = timedelta(days=1) if chunk == "day" else timedelta(hours=1)
    total_windows = int((end - start) / delta) or 1

    all_rows: list[dict] = []
    pending: list[dict] = []  # rows desde último flush
    cursor = start
    window_idx = 0

    while cursor < end:
        nxt = min(cursor + delta, end)
        try:
            articles = _fetch_window(cursor, nxt, query)
        except Exception as e:
            print(f"[gdelt]  ⚠️  {cursor.isoformat()}: {e}", flush=True)
            articles = []

        for a in articles:
            row = {
                "url": a.get("url"),
                "title": a.get("title"),
                "seendate": a.get("seendate"),
                "domain": a.get("domain"),
                "language": a.get("language"),
                "sourcecountry": a.get("sourcecountry"),
                "source": "gdelt",
            }
            all_rows.append(row)
            pending.append(row)

        window_idx += 1
        if window_idx % progress_every == 0 or cursor + delta >= end:
            pct = 100 * window_idx / total_windows
            print(
                f"[gdelt]  {cursor.date()}  {window_idx}/{total_windows} "
                f"({pct:.1f}%)  +{len(pending)} novos  total={len(all_rows)}",
                flush=True,
            )
            if flush_callback and pending:
                try:
                    flush_callback(_finalize(pending))
                except Exception as e:
                    print(f"[gdelt]  ⚠️  flush falhou: {e}", flush=True)
                pending = []

        cursor = nxt
        time.sleep(sleep_between)

    # flush final do que ainda não foi escrito
    if flush_callback and pending:
        try:
            flush_callback(_finalize(pending))
        except Exception as e:
            print(f"[gdelt]  ⚠️  flush final falhou: {e}", flush=True)

    return _finalize(all_rows)
