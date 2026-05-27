"""Snapshot reference features — gera baseline pra drift_watchdog comparar.

Lê pipeline.features.build_matrix_v2() sobre uma janela histórica congelada
(default: 2023-01-01 → 2024-06-30) e salva em data/reference_features.parquet.

Esse é o "estado normal" do mundo segundo o modelo. drift_watchdog computa
PSI das features atuais (últimos 30d) vs essa referência. PSI > 0.25 trippa.

Roda quando:
  - Antes de subir a stack pela primeira vez
  - Sempre que retrainar modelo (atualizar referência ao novo período)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

sys.path.insert(0, "/app")

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("snapshot_reference")

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
OUT = DATA_DIR / "reference_features.parquet"


def run(start: str, end: str) -> None:
    from pipeline import features  # late import (heavy)

    log.info(f"building matrix v2 between {start} and {end}…")
    try:
        mat = features.build_matrix_v2()
    except Exception:
        mat = features.build_matrix()  # fallback p/ versão pré-v2

    start_ms = int(datetime.fromisoformat(start).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(end).timestamp() * 1000)
    ts_col = "open_time" if "open_time" in mat.columns else "ts"
    ref = mat.filter((pl.col(ts_col) >= start_ms) & (pl.col(ts_col) <= end_ms))
    log.info(f"reference rows: {ref.height} (cols: {len(ref.columns)})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ref.write_parquet(OUT)
    log.info(f"saved {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2024-06-30")
    args = p.parse_args()
    run(args.start, args.end)
