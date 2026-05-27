"""Merge data/news_chunks/*.parquet em data/news_raw.parquet (dedup por id).

Uso:
    python -m pipeline.news_merge
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from pipeline import storage

DATA = Path("data")
CHUNKS_DIR = DATA / "news_chunks"
NEWS = DATA / "news_raw.parquet"


def run() -> None:
    if not CHUNKS_DIR.exists():
        print(f"[merge] {CHUNKS_DIR} não existe — nada a fazer")
        return

    chunk_files = sorted(CHUNKS_DIR.glob("*.parquet"))
    if not chunk_files:
        print("[merge] nenhum chunk encontrado")
        return

    print(f"[merge] {len(chunk_files)} chunks encontrados")
    total_before = storage.read(NEWS).height
    for f in chunk_files:
        df = pl.read_parquet(f)
        if df.is_empty():
            continue
        n = storage.upsert(NEWS, df, "id")
        print(f"  + {f.name:>50s}  {df.height:>6} rows  novos: {n}")

    total_after = storage.read(NEWS).height
    print(f"[merge] news_raw.parquet: {total_before} → {total_after}  (+{total_after - total_before})")


if __name__ == "__main__":
    run()
