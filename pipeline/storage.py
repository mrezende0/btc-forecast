"""Upsert idempotente em Parquet, com utilitário de leitura simples."""
from __future__ import annotations

from pathlib import Path

import polars as pl


def read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else pl.DataFrame()


def upsert(path: Path, new: pl.DataFrame, key: str | list[str]) -> int:
    """Mescla `new` com o Parquet em `path`, dedup pela chave, salva.

    Retorna número de linhas adicionadas (height_final - height_inicial).
    """
    if new.is_empty():
        return 0
    keys = [key] if isinstance(key, str) else list(key)
    existing = read(path)
    if existing.is_empty():
        merged = new
    else:
        merged = pl.concat([existing, new], how="vertical_relaxed")
    merged = merged.unique(subset=keys, keep="last").sort(keys)
    added = merged.height - existing.height
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(path)
    return added


def last_ts(path: Path, key: str) -> int | None:
    df = read(path)
    return None if df.is_empty() else int(df[key].max())
