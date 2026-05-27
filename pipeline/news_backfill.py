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
    output: str | None = None,
) -> None:
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = (
        datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
        if end
        else datetime.now(tz=timezone.utc)
    )

    out_path = Path(output) if output else NEWS

    print(f"[gdelt] {start_dt.date()} → {end_dt.date()}  (chunk={chunk})  out={out_path}", flush=True)

    def _normalize(df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            pl.col("url").alias("id"),
            pl.lit(None).cast(pl.Utf8).alias("body"),
            pl.lit(None).cast(pl.Utf8).alias("sentiment_src"),
            pl.lit(None).cast(pl.Utf8).alias("categories"),
        )
        keep = [
            "id", "url", "title", "body", "published_ts",
            "domain", "language", "sentiment_src", "categories", "source",
        ]
        return df.select([c for c in keep if c in df.columns])

    def _flush(batch: pl.DataFrame) -> None:
        if batch.is_empty():
            return
        n = storage.upsert(out_path, _normalize(batch), "id")
        print(f"[gdelt]  💾 flush: +{n} novos em {out_path.name}", flush=True)

    df = gdelt.fetch_range(
        start_dt,
        end_dt,
        query=query or gdelt.DEFAULT_QUERY,
        chunk=chunk,
        flush_callback=_flush,
    )

    if df.is_empty():
        print("[gdelt] nenhum artigo retornado no total", flush=True)
        return

    total = storage.read(out_path).height
    print(f"[gdelt] ✓ concluído. {out_path.name} total = {total} artigos", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--chunk", default="day", choices=["day", "hour"])
    p.add_argument("--query", default=None)
    p.add_argument("--output", default=None, help="Parquet de destino (default data/news_raw.parquet)")
    run(**vars(p.parse_args()))
