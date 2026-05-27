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

    # Agregações dinâmicas conforme colunas presentes (compat com schema legado)
    aggs = [
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
        pl.col("quote_volume").sum().alias("quote_volume"),
        pl.col("trades").sum().alias("trades"),
    ]
    if "taker_buy_volume" in df.columns:
        aggs.append(pl.col("taker_buy_volume").sum().alias("taker_buy_volume"))
    if "taker_buy_quote_volume" in df.columns:
        aggs.append(pl.col("taker_buy_quote_volume").sum().alias("taker_buy_quote_volume"))

    out = (
        df.group_by("bucket", maintain_order=True)
        .agg(*aggs)
        .rename({"bucket": "open_time"})
        .with_columns((pl.col("open_time") + interval_ms - 1).alias("close_time"))
        .sort("open_time")
    )
    return out


def resample_perp(df: pl.DataFrame, minutes: int = 240) -> pl.DataFrame:
    """Resample do perp_15m com mesmo prefix `perp_`."""
    if df.is_empty():
        return df
    interval_ms = minutes * 60 * 1000
    df = df.sort("open_time").with_columns(
        ((pl.col("open_time") // interval_ms) * interval_ms).alias("bucket")
    )
    return (
        df.group_by("bucket", maintain_order=True)
        .agg(
            pl.col("perp_open").first().alias("perp_open"),
            pl.col("perp_high").max().alias("perp_high"),
            pl.col("perp_low").min().alias("perp_low"),
            pl.col("perp_close").last().alias("perp_close"),
            pl.col("perp_volume").sum().alias("perp_volume"),
            pl.col("perp_taker_buy_volume").sum().alias("perp_taker_buy_volume"),
        )
        .rename({"bucket": "open_time"})
        .sort("open_time")
    )
