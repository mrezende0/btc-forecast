"""Feature engineering — funções puras que mapeiam Parquets brutos → matriz de features
alinhada ao grid de 15min do OHLCV.

REGRA INVIOLÁVEL: toda feature usa apenas dados conhecidos ANTES do início da vela.
Implementação: `.shift(1)` em cima de qualquer cálculo derivado de preço/volume; e
`available_at` defasado em fontes daily (macro/F&G).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

BAR_MS = 15 * 60 * 1000
BARS_PER_HOUR = 4
BARS_PER_DAY = 96
BARS_PER_WEEK = 96 * 7

DATA = Path("data")


# -------------------------------------------------------------------- helpers
def _rolling_zscore(col: str, window: int, name: str | None = None) -> pl.Expr:
    name = name or f"{col}_z{window}"
    mu = pl.col(col).rolling_mean(window_size=window)
    sd = pl.col(col).rolling_std(window_size=window)
    return ((pl.col(col) - mu) / sd).alias(name)


def _rsi(col: str, window: int = 14, name: str | None = None) -> pl.Expr:
    name = name or f"rsi_{window}"
    delta = pl.col(col).diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0)
    avg_g = gain.rolling_mean(window_size=window)
    avg_l = loss.rolling_mean(window_size=window)
    rs = avg_g / pl.when(avg_l == 0).then(1e-12).otherwise(avg_l)
    return (100 - 100 / (1 + rs)).alias(name)


def _atr(window: int = 14) -> pl.Expr:
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - pl.col("close").shift(1)).abs(),
        (pl.col("low") - pl.col("close").shift(1)).abs(),
    )
    return tr.rolling_mean(window_size=window).alias(f"atr_{window}")


# --------------------------------------------------------------- price/técnico
def add_technical(df: pl.DataFrame) -> pl.DataFrame:
    out = df.sort("open_time").with_columns(
        pl.col("close").pct_change().alias("ret_1"),
        pl.col("close").pct_change(BARS_PER_HOUR).alias("ret_1h"),
        pl.col("close").pct_change(BARS_PER_HOUR * 4).alias("ret_4h"),
        pl.col("close").pct_change(BARS_PER_DAY).alias("ret_1d"),
        pl.col("close").pct_change(BARS_PER_WEEK).alias("ret_1w"),
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("logret_1"),
    )

    out = out.with_columns(
        # Vol realizada
        pl.col("logret_1").rolling_std(window_size=BARS_PER_HOUR * 4).alias("rv_4h"),
        pl.col("logret_1").rolling_std(window_size=BARS_PER_DAY).alias("rv_1d"),
        pl.col("logret_1").rolling_std(window_size=BARS_PER_WEEK).alias("rv_1w"),
        # ATR
        _atr(14),
        # RSI multi-TF (usa close em escala 15m, depois proxy 1h/4h via window maior)
        _rsi("close", 14, "rsi_15m"),
        _rsi("close", 56, "rsi_1h"),  # ~14 barras de 1h
        _rsi("close", 224, "rsi_4h"),
    )

    out = out.with_columns(
        # MAs e distância
        pl.col("close").rolling_mean(window_size=BARS_PER_DAY * 7).alias("ma_7d"),
        pl.col("close").rolling_mean(window_size=BARS_PER_DAY * 30).alias("ma_30d"),
        pl.col("close").rolling_mean(window_size=BARS_PER_DAY * 90).alias("ma_90d"),
    )
    out = out.with_columns(
        (pl.col("close") / pl.col("ma_7d") - 1).alias("dist_ma_7d"),
        (pl.col("close") / pl.col("ma_30d") - 1).alias("dist_ma_30d"),
        (pl.col("close") / pl.col("ma_90d") - 1).alias("dist_ma_90d"),
    )

    # Volume Z-score
    out = out.with_columns(
        _rolling_zscore("volume", BARS_PER_DAY * 7, "vol_z7d"),
        _rolling_zscore("volume", BARS_PER_DAY * 30, "vol_z30d"),
    )

    # Bollinger position (close vs banda) — 20 barras de 15m = 5h
    bb_window = 20
    out = out.with_columns(
        pl.col("close").rolling_mean(window_size=bb_window).alias("_bb_mid"),
        pl.col("close").rolling_std(window_size=bb_window).alias("_bb_sd"),
    )
    out = out.with_columns(
        ((pl.col("close") - pl.col("_bb_mid")) / (pl.col("_bb_sd") * 2)).alias("bb_pos")
    ).drop(["_bb_mid", "_bb_sd"])

    return out


# -------------------------------------------------------------- derivativos
def add_funding(df: pl.DataFrame, funding: pl.DataFrame) -> pl.DataFrame:
    """As-of join backward — funding `t` é o mais recente publicado <= bar open_time."""
    f = funding.sort("funding_time").with_columns(
        pl.col("funding_rate").alias("funding"),
        _rolling_zscore("funding_rate", 90, "funding_z90"),  # ~30d (3 funding/dia)
        pl.col("funding_rate").ewm_mean(span=24).alias("funding_ema8d"),
    )
    out = df.sort("open_time").join_asof(
        f.select(["funding_time", "funding", "funding_z90", "funding_ema8d"]),
        left_on="open_time",
        right_on="funding_time",
        strategy="backward",
    )
    # horas até próximo funding (a cada 8h) — feature de calendário derivativos
    out = out.with_columns(
        (
            ((pl.col("open_time") - pl.col("funding_time")) / (1000 * 60 * 60)).cast(pl.Float64)
        ).alias("hours_since_funding"),
    )
    return out.drop("funding_time")


# ------------------------------------------------------------------- macro
def add_macro(df: pl.DataFrame, macro: pl.DataFrame) -> pl.DataFrame:
    """Daily fonte. available_at = dia D+1 06:00 UTC (após publishes EOD).

    Implementação: shift de 1 dia + as-of join backward na data calculada.
    """
    m = (
        macro.sort("date")
        .with_columns(
            pl.col("dxy").pct_change().alias("dxy_ret"),
            pl.col("vix").pct_change().alias("vix_ret"),
            pl.col("spx").pct_change().alias("spx_ret"),
        )
        .with_columns(
            _rolling_zscore("dxy", 30, "dxy_z30"),
            _rolling_zscore("vix", 30, "vix_z30"),
        )
    )
    # available_at = date + 1 dia 06:00 UTC
    m = m.with_columns(
        (pl.col("date").cast(pl.Datetime) + pl.duration(days=1, hours=6))
        .dt.timestamp("ms")
        .alias("available_at_ms")
    ).drop("date")
    m = m.select(["available_at_ms", "dxy_ret", "vix_ret", "spx_ret", "dxy_z30", "vix_z30"]).sort(
        "available_at_ms"
    )

    return df.sort("open_time").join_asof(
        m,
        left_on="open_time",
        right_on="available_at_ms",
        strategy="backward",
    ).drop("available_at_ms")


# ---------------------------------------------------------------- sentimento
def add_sentiment_fg(df: pl.DataFrame, fg: pl.DataFrame) -> pl.DataFrame:
    """F&G publica diariamente. available_at = mesma data 00:30 UTC.

    Pra simplificar e ser conservador, usamos data anterior (lag 1 dia).
    """
    f = (
        fg.sort("date")
        .with_columns(
            pl.col("fg_value").alias("fg"),
            _rolling_zscore("fg_value", 30, "fg_z30"),
            (pl.col("fg_value") - pl.col("fg_value").shift(7)).alias("fg_chg7"),
        )
        .with_columns(
            (pl.col("date").cast(pl.Datetime) + pl.duration(days=1))
            .dt.timestamp("ms")
            .alias("available_at_ms")
        )
        .drop("date")
    )
    f = f.select(["available_at_ms", "fg", "fg_z30", "fg_chg7"]).sort("available_at_ms")

    return df.sort("open_time").join_asof(
        f,
        left_on="open_time",
        right_on="available_at_ms",
        strategy="backward",
    ).drop("available_at_ms")


def add_sentiment_news(df: pl.DataFrame, sentiment_daily: pl.DataFrame) -> pl.DataFrame:
    """Sentiment agregado de notícias (após FinBERT scoring + sentiment_agg).

    Lag 1 dia também — usamos média do dia anterior.
    """
    if sentiment_daily.is_empty():
        return df

    s = (
        sentiment_daily.sort("date")
        .with_columns(
            _rolling_zscore("news_count", 30, "news_count_z30"),
            _rolling_zscore("net_sentiment", 30, "net_sentiment_z30"),
        )
        .with_columns(
            (pl.col("date").cast(pl.Datetime) + pl.duration(days=1))
            .dt.timestamp("ms")
            .alias("available_at_ms")
        )
        .drop("date")
    )
    cols = ["available_at_ms", "news_count", "net_sentiment", "news_count_z30", "net_sentiment_z30"]
    s = s.select([c for c in cols if c in s.columns]).sort("available_at_ms")

    return df.sort("open_time").join_asof(
        s,
        left_on="open_time",
        right_on="available_at_ms",
        strategy="backward",
    ).drop("available_at_ms")


# ----------------------------------------------------------------- calendário
def add_calendar(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.from_epoch(pl.col("open_time") // 1000, time_unit="s").alias("_ts")
    ).with_columns(
        pl.col("_ts").dt.hour().alias("hour"),
        pl.col("_ts").dt.weekday().alias("dow"),
    ).with_columns(
        (np.sin(2 * np.pi * pl.col("hour") / 24)).alias("hour_sin"),
        (np.cos(2 * np.pi * pl.col("hour") / 24)).alias("hour_cos"),
        ((pl.col("hour") >= 12) & (pl.col("hour") <= 21)).cast(pl.Int8).alias("is_us_session"),
        (pl.col("dow") >= 6).cast(pl.Int8).alias("is_weekend"),
    ).drop("_ts")


# ------------------------------------------------------- lag invariável final
LAG_SAFE_EXCLUDE = {
    "open_time", "close_time",
    # calendário pode usar hora atual (a vela está em formação? NÃO, é o open dela)
    "hour", "dow", "hour_sin", "hour_cos", "is_us_session", "is_weekend",
    # alvos OHLCV — não são features, apenas suporte
    "open", "high", "low", "close", "volume", "quote_volume", "trades",
}


def apply_lag(df: pl.DataFrame, lag: int = 1) -> pl.DataFrame:
    """Defasa TODAS as colunas de features (não-excluídas) por `lag` velas.

    Garante que, ao abrir a vela t, o modelo só vê features computadas até t-1.
    """
    feat_cols = [c for c in df.columns if c not in LAG_SAFE_EXCLUDE]
    return df.with_columns([pl.col(c).shift(lag).alias(c) for c in feat_cols])


# ------------------------------------------------------------------ pipeline
def build(
    ohlcv: pl.DataFrame,
    funding: pl.DataFrame,
    macro: pl.DataFrame,
    fg: pl.DataFrame,
    sentiment_daily: pl.DataFrame | None = None,
    lag: int = 1,
) -> pl.DataFrame:
    df = ohlcv.sort("open_time")
    df = add_technical(df)
    df = add_funding(df, funding)
    df = add_macro(df, macro)
    df = add_sentiment_fg(df, fg)
    if sentiment_daily is not None and not sentiment_daily.is_empty():
        df = add_sentiment_news(df, sentiment_daily)
    df = add_calendar(df)
    df = apply_lag(df, lag=lag)
    return df


def build_from_parquets(lag: int = 1) -> pl.DataFrame:
    ohlcv = pl.read_parquet(DATA / "ohlcv_15m.parquet")
    funding = pl.read_parquet(DATA / "funding.parquet")
    macro = pl.read_parquet(DATA / "macro_daily.parquet")
    fg = pl.read_parquet(DATA / "fg_daily.parquet")
    sd_path = DATA / "sentiment_daily.parquet"
    sd = pl.read_parquet(sd_path) if sd_path.exists() else pl.DataFrame()
    return build(ohlcv, funding, macro, fg, sd, lag=lag)
