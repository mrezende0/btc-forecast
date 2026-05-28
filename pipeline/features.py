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
    # MAs absolutas: nível em USD, escalam com preço → drift severo em bull market.
    # Usadas apenas como insumo intermediário pra dist_ma_* (scale-free) e is_uptrend_*.
    "ma_7d", "ma_30d", "ma_90d",
    # Demais features scale-dependent: substituídas por equivalentes normalizados.
    # atr_14 → mantida no df (necessária pra stop/triple-barrier), só fora do modelo. Use rv_* como vol feature.
    # taker_buy_volume(_quote) → use taker_buy_ratio (já scale-free).
    # news_count → use news_count_z30 (rolling z-score já existe).
    "atr_14", "taker_buy_volume", "taker_buy_quote_volume", "news_count",
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


# ============================================================================
# v2 — timeframe configurável + interactions + regime
# ============================================================================
def add_technical_tf(df: pl.DataFrame, bph: int, bpd: int, bpw: int) -> pl.DataFrame:
    """Versão timeframe-aware. `bph` = bars per hour (pode ser fracionário arredondado)."""
    out = df.sort("open_time").with_columns(
        pl.col("close").pct_change().alias("ret_1"),
        pl.col("close").pct_change(max(1, bph)).alias("ret_1h"),
        pl.col("close").pct_change(max(1, bph * 4)).alias("ret_4h"),
        pl.col("close").pct_change(bpd).alias("ret_1d"),
        pl.col("close").pct_change(bpw).alias("ret_1w"),
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("logret_1"),
    )
    out = out.with_columns(
        pl.col("logret_1").rolling_std(window_size=max(2, bph * 4)).alias("rv_4h"),
        pl.col("logret_1").rolling_std(window_size=bpd).alias("rv_1d"),
        pl.col("logret_1").rolling_std(window_size=bpw).alias("rv_1w"),
        _atr(14),
        _rsi("close", 14, "rsi_14"),
        _rsi("close", max(2, bpd // 2), "rsi_halfday"),
        _rsi("close", bpd, "rsi_1d"),
    )
    out = out.with_columns(
        pl.col("close").rolling_mean(window_size=bpd * 7).alias("ma_7d"),
        pl.col("close").rolling_mean(window_size=bpd * 30).alias("ma_30d"),
        pl.col("close").rolling_mean(window_size=bpd * 90).alias("ma_90d"),
    )
    out = out.with_columns(
        (pl.col("close") / pl.col("ma_7d") - 1).alias("dist_ma_7d"),
        (pl.col("close") / pl.col("ma_30d") - 1).alias("dist_ma_30d"),
        (pl.col("close") / pl.col("ma_90d") - 1).alias("dist_ma_90d"),
    )
    out = out.with_columns(
        _rolling_zscore("volume", bpd * 7, "vol_z7d"),
        _rolling_zscore("volume", bpd * 30, "vol_z30d"),
    )
    bb_w = max(4, bpd // 2)
    out = out.with_columns(
        pl.col("close").rolling_mean(window_size=bb_w).alias("_bb_mid"),
        pl.col("close").rolling_std(window_size=bb_w).alias("_bb_sd"),
    )
    out = out.with_columns(
        ((pl.col("close") - pl.col("_bb_mid")) / (pl.col("_bb_sd") * 2)).alias("bb_pos")
    ).drop(["_bb_mid", "_bb_sd"])
    return out


def add_interactions(df: pl.DataFrame) -> pl.DataFrame:
    """Cross-feature combinations baseadas em hipóteses da EDA.

    - funding × ret_1d: funding alto + price up = correção iminente?
    - vix_ret × spx_ret: risk-off uniforme vs divergência
    - rsi × bb_pos: confirmação multi-sinal de extremo
    - dxy_z × dist_ma_30d: força dólar + distância de trend
    - fg_chg7 × ret_1w: sentimento mudando + price action
    """
    return df.with_columns(
        (pl.col("funding") * pl.col("ret_1d")).alias("ix_funding_ret1d"),
        (pl.col("vix_ret") * pl.col("spx_ret")).alias("ix_vix_spx"),
        (pl.col("rsi_14") * pl.col("bb_pos")).alias("ix_rsi_bb"),
        (pl.col("dxy_z30") * pl.col("dist_ma_30d")).alias("ix_dxy_dist30"),
        (pl.col("fg_chg7") * pl.col("ret_1w")).alias("ix_fgchg_ret1w"),
        (pl.col("funding_z90") * pl.col("rv_1d")).alias("ix_funding_vol"),
    )


def add_regime(df: pl.DataFrame) -> pl.DataFrame:
    """Indicadores de regime — bull/bear/chop e vol baixa/alta.

    - is_uptrend_short: MA_7d > MA_30d
    - is_uptrend_long:  MA_30d > MA_90d
    - trend_strength:   abs(dist_ma_30d)
    - vol_high:         rv_1d acima do percentil 75 rolling 30d
    - drawdown_from_high: distância da máxima 30d
    """
    return df.with_columns(
        (pl.col("ma_7d") > pl.col("ma_30d")).cast(pl.Int8).alias("is_uptrend_short"),
        (pl.col("ma_30d") > pl.col("ma_90d")).cast(pl.Int8).alias("is_uptrend_long"),
        pl.col("dist_ma_30d").abs().alias("trend_strength"),
        (
            pl.col("close") / pl.col("close").rolling_max(window_size=180) - 1
        ).alias("drawdown_30d"),
    ).with_columns(
        (
            pl.col("rv_1d") > pl.col("rv_1d").rolling_quantile(window_size=180, quantile=0.75)
        ).cast(pl.Int8).alias("vol_high"),
    )


def add_flow(df: pl.DataFrame, perp: pl.DataFrame | None = None, bpd: int = 6) -> pl.DataFrame:
    """Features de microestrutura: taker_buy_ratio, OFI proxy, basis (spot vs perp).

    REGRA: todas computadas a partir de vela JÁ FECHADA → seguro (apply_lag final
    move 1 vela atrás).
    """
    out = df
    if "taker_buy_volume" in df.columns and "volume" in df.columns:
        out = out.with_columns(
            pl.when(pl.col("volume") > 0)
              .then(pl.col("taker_buy_volume") / pl.col("volume"))
              .otherwise(0.5)
              .alias("taker_buy_ratio"),
        )
        # OFI proxy: (taker_buy - taker_sell) / volume = 2*ratio - 1, em [-1, +1]
        out = out.with_columns(
            (2 * pl.col("taker_buy_ratio") - 1).alias("ofi_proxy"),
            _rolling_zscore("taker_buy_ratio", bpd * 7, "taker_buy_ratio_z7d"),
            _rolling_zscore("taker_buy_ratio", bpd * 30, "taker_buy_ratio_z30d"),
        )

    # Basis (spot vs perp): perp_close / spot_close - 1. Positivo = perp prêmio (longs aquecidos).
    if perp is not None and not perp.is_empty():
        perp_use = perp.select(["open_time", "perp_close"])
        if "perp_taker_buy_volume" in perp.columns and "perp_volume" in perp.columns:
            perp_use = perp.select(["open_time", "perp_close", "perp_taker_buy_volume", "perp_volume"])
        out = out.sort("open_time").join(perp_use, on="open_time", how="left")
        out = out.with_columns(
            (pl.col("perp_close") / pl.col("close") - 1).alias("basis"),
        )
        out = out.with_columns(
            _rolling_zscore("basis", bpd * 7, "basis_z7d"),
            _rolling_zscore("basis", bpd * 30, "basis_z30d"),
        )
        if "perp_taker_buy_volume" in out.columns:
            out = out.with_columns(
                pl.when(pl.col("perp_volume") > 0)
                  .then(pl.col("perp_taker_buy_volume") / pl.col("perp_volume"))
                  .otherwise(0.5)
                  .alias("perp_taker_buy_ratio"),
            )
            # diff de agressão: longs perp mais agressivos que spot?
            if "taker_buy_ratio" in out.columns:
                out = out.with_columns(
                    (pl.col("perp_taker_buy_ratio") - pl.col("taker_buy_ratio")).alias("flow_div_perp_spot"),
                )
        # drop colunas intermediárias que não devem ser features
        for c in ("perp_close", "perp_taker_buy_volume", "perp_volume"):
            if c in out.columns:
                out = out.drop(c)
    return out


def build_v2(
    ohlcv: pl.DataFrame,
    funding: pl.DataFrame,
    macro: pl.DataFrame,
    fg: pl.DataFrame,
    sentiment_daily: pl.DataFrame | None = None,
    perp: pl.DataFrame | None = None,
    timeframe_min: int = 240,  # 4h default
    lag: int = 1,
) -> pl.DataFrame:
    """Pipeline completa em timeframe arbitrário, com interactions + regime + flow."""
    from pipeline import resample

    if timeframe_min != 15:
        ohlcv = resample.resample_ohlcv(ohlcv, minutes=timeframe_min)
        if perp is not None and not perp.is_empty():
            perp = resample.resample_perp(perp, minutes=timeframe_min)

    bph = max(1, round(60 / timeframe_min))
    bpd = round(60 * 24 / timeframe_min)
    bpw = bpd * 7

    df = ohlcv.sort("open_time")
    df = add_technical_tf(df, bph=bph, bpd=bpd, bpw=bpw)
    df = add_funding(df, funding)
    df = add_macro(df, macro)
    df = add_sentiment_fg(df, fg)
    if sentiment_daily is not None and not sentiment_daily.is_empty():
        df = add_sentiment_news(df, sentiment_daily)
    df = add_calendar(df)
    df = add_interactions(df)
    df = add_regime(df)
    df = add_flow(df, perp=perp, bpd=bpd)
    df = apply_lag(df, lag=lag)
    return df


def build_v2_from_parquets(timeframe_min: int = 240, lag: int = 1, asset: str = "BTC") -> pl.DataFrame:
    """Build features from parquets do ativo especificado. Default BTC (legado)."""
    from pipeline import assets as _assets
    cfg = _assets.get(asset)
    ohlcv = pl.read_parquet(cfg["ohlcv"])
    funding = pl.read_parquet(cfg["funding"])
    macro = pl.read_parquet(DATA / "macro_daily.parquet")
    fg = pl.read_parquet(DATA / "fg_daily.parquet")
    sd_path = DATA / "sentiment_daily.parquet"
    sd = pl.read_parquet(sd_path) if sd_path.exists() else pl.DataFrame()
    perp = pl.read_parquet(cfg["perp"]) if cfg["perp"].exists() else pl.DataFrame()
    return build_v2(ohlcv, funding, macro, fg, sd, perp=perp, timeframe_min=timeframe_min, lag=lag)
