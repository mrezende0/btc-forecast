"""Fetcher CoinDesk Data API — notícias agregadas de ~100 portais cripto.

Free tier: ~5.000 calls/mês. Cada call retorna até 100 manchetes.
Schema do response oficial documentado em developers.coindesk.com/documentation
"""
from __future__ import annotations

import os
import time

import polars as pl
import requests

API = "https://data-api.coindesk.com/news/v1/article/list"
LIMIT = 100  # máx por request


def _key() -> str:
    k = os.environ.get("COINDESK_API_KEY", "").strip()
    if not k:
        raise RuntimeError("Falta env COINDESK_API_KEY. Cadastrar em developers.coindesk.com")
    return k


def _request(to_ts: int | None = None, lang: str = "EN") -> list[dict]:
    params = {"lang": lang, "limit": LIMIT}
    if to_ts is not None:
        params["to_ts"] = to_ts
    headers = {"authorization": f"Apikey {_key()}"}
    r = requests.get(API, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    payload = r.json()
    # Resposta pode vir como {Data: [...]} ou {data: [...]} dependendo do endpoint.
    return payload.get("Data") or payload.get("data") or []


def _parse(items: list[dict]) -> list[dict]:
    out: list[dict] = []
    for a in items:
        source_data = a.get("SOURCE_DATA") or {}
        cats = a.get("CATEGORY_DATA") or []
        out.append(
            {
                "id": a.get("ID") or a.get("GUID"),
                "url": a.get("URL"),
                "title": a.get("TITLE"),
                "body": a.get("BODY"),
                "published_ts": int(a.get("PUBLISHED_ON") or 0),
                "domain": source_data.get("NAME"),
                "language": a.get("LANG", "EN"),
                "sentiment_src": a.get("SENTIMENT"),  # POSITIVE|NEUTRAL|NEGATIVE
                "categories": ",".join(c.get("CATEGORY", "") for c in cats),
                "source": "coindesk",
            }
        )
    return out


def fetch_latest(pages: int = 1, lang: str = "EN") -> pl.DataFrame:
    """Modo incremental: puxa as N páginas mais recentes (default 1 = 100 artigos).

    Pra incremental no Actions, 1 página basta — você dedup contra o Parquet existente.
    """
    rows: list[dict] = []
    to_ts: int | None = None
    for _ in range(pages):
        items = _request(to_ts=to_ts, lang=lang)
        if not items:
            break
        parsed = _parse(items)
        rows.extend(parsed)
        # próximo cursor: artigo mais antigo da página
        oldest = min((p["published_ts"] for p in parsed if p["published_ts"]), default=0)
        if not oldest:
            break
        to_ts = oldest - 1
        time.sleep(0.5)

    if not rows:
        return pl.DataFrame()

    return (
        pl.DataFrame(rows)
        .drop_nulls(subset=["id", "title", "published_ts"])
        .unique(subset=["id"], keep="first")
        .sort("published_ts")
    )


def fetch_history(
    until_ts: int,
    lang: str = "EN",
    max_pages: int = 50,
    sleep_between: float = 0.5,
) -> pl.DataFrame:
    """Pagina retroativamente até cobrir histórico desejado (best-effort).

    Free tier cobre ~últimos meses no máximo — pra anos, usar GDELT como backfill.
    """
    rows: list[dict] = []
    to_ts: int | None = None
    for _ in range(max_pages):
        items = _request(to_ts=to_ts, lang=lang)
        if not items:
            break
        parsed = _parse(items)
        rows.extend(parsed)
        oldest = min((p["published_ts"] for p in parsed if p["published_ts"]), default=0)
        if not oldest or oldest <= until_ts:
            break
        to_ts = oldest - 1
        time.sleep(sleep_between)

    if not rows:
        return pl.DataFrame()

    return (
        pl.DataFrame(rows)
        .filter(pl.col("published_ts") >= until_ts)
        .drop_nulls(subset=["id", "title", "published_ts"])
        .unique(subset=["id"], keep="first")
        .sort("published_ts")
    )
