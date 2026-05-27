"""Resample OHLCV de 15m pra timeframe maior. OHLC + volume agregados corretamente."""
from __future__ import annotations

import polars as pl


def resample_ohlcv(df: pl.DataFrame, minutes: int = 240) -> pl.DataFrame:
    """Agrega velas 15m em janelas fixas de `minutes`.

    Default 240 = 4h. Mantém schema do OHLCV original.
    """
    interval_ms = minutes * 60 * 1000
    df = df.sort("open_time").with_columns(
        ((pl.col("open_time") // interval_ms) * interval_ms).alias("bucket")
    )

    out = (
        df.group_by("bucket", maintain_order=True)
        .agg(
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("quote_volume").sum().alias("quote_volume"),
            pl.col("trades").sum().alias("trades"),
        )
        .rename({"bucket": "open_time"})
        .with_columns((pl.col("open_time") + interval_ms - 1).alias("close_time"))
        .sort("open_time")
    )
    return out
