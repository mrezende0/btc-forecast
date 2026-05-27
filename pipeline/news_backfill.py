"""Backfill histórico de notícias via GDELT — rodar local uma vez.

Uso:
    python -m pipeline.news_backfill --start 2021-01-01 --end 2025-01-01
    python -m pipeline.news_backfill --start 2024-12-01 --chunk hour   # densidade alta
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from pipeline import gdelt, storage

DATA = Path("data")
NEWS = DATA / "news_raw.parquet"


def run(
    start: str = "2021-01-01",
    end: str | None = None,
    chunk: str = "day",
    query: str | None = None,
) -> None:
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = (
        datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
        if end
        else datetime.now(tz=timezone.utc)
    )

    print(f"[gdelt] {start_dt.date()} → {end_dt.date()}  (chunk={chunk})")
    df = gdelt.fetch_range(
        start_dt,
        end_dt,
        query=query or gdelt.DEFAULT_QUERY,
        chunk=chunk,
    )

    if df.is_empty():
        print("[gdelt] nenhum artigo retornado")
        return

    # Normaliza schema com CoinDesk (gera colunas faltantes pra merge)
    df = df.with_columns(
        pl.lit(None).cast(pl.Utf8).alias("id"),
        pl.lit(None).cast(pl.Utf8).alias("body"),
        pl.lit(None).cast(pl.Utf8).alias("sentiment_src"),
        pl.lit(None).cast(pl.Utf8).alias("categories"),
    )
    # ID estável a partir da URL (GDELT não tem ID próprio)
    df = df.with_columns(pl.col("url").alias("id"))

    keep = [
        "id", "url", "title", "body", "published_ts",
        "domain", "language", "sentiment_src", "categories", "source",
    ]
    df = df.select([c for c in keep if c in df.columns])

    n = storage.upsert(NEWS, df, "id")
    print(f"[gdelt] +{n} artigos novos (total {storage.read(NEWS).height})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--chunk", default="day", choices=["day", "hour"])
    p.add_argument("--query", default=None)
    run(**vars(p.parse_args()))
