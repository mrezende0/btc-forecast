"""Microstructure features — proposta quant (Cont-Kukanov, Easley-LdP-O'Hara, Stoikov).

Stub importável. Adiciona OFI proxy, taker buy ratio, CVD, basis perp-spot, OI delta
e long/short ratio sobre data/ohlcv_15m.parquet. Endpoints Binance públicos, custo zero.

REGRA: toda feature passa por `.shift(1)` no apply_lag final do pipeline.features.

Refs principais:
  - Cont, Kukanov & Stoikov (2014) — OFI ~ price change (JFinEcon)
  - Easley, López de Prado & O'Hara (2012) — VPIN flow toxicity
  - Stoikov (2018) — microprice as martingale fair price
  - Anastasopoulos & Gradojevic (EFMA 2025) — order flow → crypto returns

Status: PROPOSAL. Endpoints/parses marcados TODO precisam ser implementados antes
de plugar no exp_ensemble.py.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

DATA = Path("data")
BAR_MS = 15 * 60 * 1000
BARS_PER_DAY = 96
BARS_PER_WEEK = BARS_PER_DAY * 7


# =========================================================================
# 1. Data pulls — endpoints Binance públicos (rate limit 1000 req / 5min)
# =========================================================================

def fetch_klines_with_taker(start_ms: int, end_ms: int | None = None) -> pl.DataFrame:
    """TODO: refactor pipeline/binance.py:fetch_klines para preservar campos 9-10.

    Resposta kline tem 12 campos. Atualmente parse extrai 0-8.
    Falta:
      - r[9]  : taker_buy_base_asset_volume  (BTC comprado em market orders)
      - r[10] : taker_buy_quote_asset_volume (USDT correspondente)

    Custo: ZERO. Dado já vem no payload, só não está sendo lido.
    Endpoint: https://data-api.binance.vision/api/v3/klines (spot) ou
              https://fapi.binance.com/fapi/v1/klines       (perp futures — recomendado)

    Para microestrutura é melhor usar PERP futures klines — agressão lá é o que move
    o preço dado o volume relativo perp/spot ~7:1 em BTC.
    """
    raise NotImplementedError("Refit binance.py — preservar campos 9-10 do kline")


def fetch_open_interest_hist(symbol: str = "BTCUSDT", period: str = "15m") -> pl.DataFrame:
    """TODO: GET /futures/data/openInterestHist
    Docs: https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics

    Params: symbol, period (5m/15m/30m/1h/2h/4h/6h/12h/1d), limit (max 500), startTime/endTime.
    Apenas últimos 30 dias disponíveis — backfill iterativo em chunks de 30d.

    Schema esperado:
      timestamp (ms) | sumOpenInterest (BTC) | sumOpenInterestValue (USDT)
    """
    raise NotImplementedError("Implementar paginated backfill de openInterestHist")


def fetch_premium_index(symbol: str = "BTCUSDT") -> pl.DataFrame:
    """TODO: GET /fapi/v1/premiumIndex — mark price, index price, last funding, next funding.
    Docs: https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Mark-Price

    Para HISTÓRICO usar markPriceKlines: GET /fapi/v1/markPriceKlines (interval 15m).
    Basis = (mark - index) / index → série temporal de 15m perfeita.
    """
    raise NotImplementedError("Implementar markPriceKlines + indexPriceKlines backfill")


def fetch_top_long_short_position_ratio(symbol: str = "BTCUSDT", period: str = "15m") -> pl.DataFrame:
    """TODO: GET /futures/data/topLongShortPositionRatio
    Docs: https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Top-Trader-Long-Short-Ratio

    Schema: timestamp | longShortRatio | longAccount | shortAccount
    Sinal de positioning de top traders — leading indicator de squeeze.
    """
    raise NotImplementedError("Implementar topLongShortPositionRatio backfill")


# =========================================================================
# 2. Features — todas baseadas em OHLCV + taker_buy_base disponíveis na vela
# =========================================================================

def _rolling_zscore(col: str, window: int, name: str | None = None) -> pl.Expr:
    name = name or f"{col}_z{window}"
    mu = pl.col(col).rolling_mean(window_size=window)
    sd = pl.col(col).rolling_std(window_size=window)
    return ((pl.col(col) - mu) / sd).alias(name)


def add_taker_flow(df: pl.DataFrame) -> pl.DataFrame:
    """Taker Buy Ratio + OFI proxy + CVD + CVD divergence.

    Requer colunas: taker_buy_base, volume, close, ret_1.

    Features adicionadas:
      - taker_buy_ratio       : taker_buy / volume  ∈ [0, 1], 0.5 = neutro
      - taker_buy_ratio_z7d   : Z-score rolling 7d
      - taker_buy_ratio_z30d  : Z-score rolling 30d
      - ofi_proxy             : (2*taker_buy - volume) / volume  ∈ [-1, 1]
                                = Cont-Kukanov OFI normalizado por barra
      - cvd                   : cumsum(taker_buy - taker_sell) — net buying pressure
      - cvd_chg_1d            : Δcvd em 1 dia (96 barras 15m)
      - cvd_div_1d            : ret_1d - normalized(cvd_chg_1d) — divergence signal
                                positivo = preço sobe sem fluxo (bearish div)
                                negativo = preço cai com fluxo comprador (bullish div)

    Ref: Cont/Kukanov/Stoikov 2014; Anastasopoulos & Gradojevic 2025.
    """
    if "taker_buy_base" not in df.columns:
        raise KeyError(
            "taker_buy_base ausente — rodar fetch_klines_with_taker primeiro. "
            "Esse campo é o índice 9 do kline response Binance."
        )

    out = df.sort("open_time").with_columns(
        (pl.col("taker_buy_base") / pl.col("volume")).alias("taker_buy_ratio"),
        ((2 * pl.col("taker_buy_base") - pl.col("volume")) / pl.col("volume")).alias("ofi_proxy"),
        (2 * pl.col("taker_buy_base") - pl.col("volume")).alias("_signed_vol"),
    )
    out = out.with_columns(
        pl.col("_signed_vol").cum_sum().alias("cvd"),
        _rolling_zscore("taker_buy_ratio", BARS_PER_WEEK, "taker_buy_ratio_z7d"),
        _rolling_zscore("taker_buy_ratio", BARS_PER_DAY * 30, "taker_buy_ratio_z30d"),
        _rolling_zscore("ofi_proxy", BARS_PER_WEEK, "ofi_proxy_z7d"),
    )
    out = out.with_columns(
        (pl.col("cvd") - pl.col("cvd").shift(BARS_PER_DAY)).alias("cvd_chg_1d"),
    )
    # Divergence: normalizar cvd_chg para escala de retorno e subtrair
    # TODO: trocar normalização naive por ranking percentil dentro da janela
    out = out.with_columns(
        (
            pl.col("close").pct_change(BARS_PER_DAY)
            - (pl.col("cvd_chg_1d") / pl.col("volume").rolling_sum(BARS_PER_DAY))
        ).alias("cvd_div_1d"),
    )
    return out.drop("_signed_vol")


def add_microprice(df: pl.DataFrame) -> pl.DataFrame:
    """Microprice approximation a partir de OHLCV agregado.

    Stoikov 2018 define microprice com bid/ask/imbalance. Sem L2 não temos isso direto.
    PROXY 15m: ponderar close pela direção da pressão (taker_buy_ratio como surrogate
    do imbalance). microprice_drift = (microprice - close) / close.

    TODO: substituir por L2 real quando tiver depth feed (CCXT depth subscription
    ou Binance @depth20 stream + agregação a 15m).

    Features:
      - microprice_proxy     : close * (1 + alpha * (taker_buy_ratio - 0.5))
      - microprice_drift     : (microprice_proxy - close) / close
      - microprice_drift_z   : Z-score rolling 7d
    """
    alpha = 0.001  # escala — calibrar via grid
    out = df.with_columns(
        (pl.col("close") * (1 + alpha * (pl.col("taker_buy_ratio") - 0.5))).alias("microprice_proxy"),
    ).with_columns(
        ((pl.col("microprice_proxy") - pl.col("close")) / pl.col("close")).alias("microprice_drift"),
    ).with_columns(
        _rolling_zscore("microprice_drift", BARS_PER_WEEK, "microprice_drift_z7d"),
    )
    return out


def add_vpin(df: pl.DataFrame, bucket_volume: float | None = None, n_buckets: int = 50) -> pl.DataFrame:
    """VPIN aproximado em volume clock — Easley/LdP/O'Hara 2012.

    Algoritmo:
      1. Definir bucket de volume V (default = mediana de 1d-volume).
      2. Acumular volumes até completar bucket. Em cada bucket, |buy - sell| / V.
      3. VPIN = média móvel dos últimos n_buckets desses ratios.

    Implementação polars (TODO): hoje só temos volume agregado 15m. VPIN puro precisa
    de tick-by-tick ou pelo menos barras de volume constante. Como first cut:
      vpin_proxy = rolling_mean(|ofi_proxy|, window=n_buckets)
    captura toxicidade média da janela.

    Features:
      - vpin_proxy_50  : média móvel 50 barras de |OFI|  (~12h de toxicidade média)
      - vpin_proxy_200 : 200 barras (~2 dias)
      - vpin_spike     : vpin_proxy_50 > rolling_quantile(0.9, 30d)
    """
    out = df.with_columns(
        pl.col("ofi_proxy").abs().rolling_mean(window_size=50).alias("vpin_proxy_50"),
        pl.col("ofi_proxy").abs().rolling_mean(window_size=200).alias("vpin_proxy_200"),
    ).with_columns(
        (
            pl.col("vpin_proxy_50")
            > pl.col("vpin_proxy_50").rolling_quantile(window_size=BARS_PER_DAY * 30, quantile=0.9)
        ).cast(pl.Int8).alias("vpin_spike"),
    )
    return out


def add_basis_oi(
    df: pl.DataFrame,
    basis: pl.DataFrame,
    oi: pl.DataFrame,
    long_short: pl.DataFrame,
) -> pl.DataFrame:
    """As-of backward joins de basis perp-spot, OI delta e long/short ratio.

    basis schema (esperado):    open_time | mark_close | index_close
    oi schema:                  timestamp | sum_oi_btc | sum_oi_usdt
    long_short schema:          timestamp | long_short_ratio | long_pct | short_pct

    Features adicionadas:
      - basis_bps              : 10000 * (mark - index) / index  — basis em bps
      - basis_z30d             : Z-score 30d
      - oi_btc                 : sum_oi em BTC
      - oi_chg_1d_pct          : %Δ OI em 1d (96 barras)
      - oi_chg_4h_pct          : %Δ OI em 4h (16 barras)
      - oi_ret_div             : sign(ret_4h) != sign(oi_chg_4h_pct)  (1 = divergence)
                                 = preço sobe + OI cai → short cover; preço sobe + OI sobe → new longs (squeeze risk)
      - long_short_ratio_top   : top trader long/short ratio
      - long_short_z7d         : Z-score 7d
    """
    out = df.sort("open_time")

    # basis
    if not basis.is_empty():
        b = basis.sort("open_time").with_columns(
            (10000 * (pl.col("mark_close") - pl.col("index_close")) / pl.col("index_close")).alias("basis_bps"),
        ).with_columns(
            _rolling_zscore("basis_bps", BARS_PER_DAY * 30, "basis_z30d"),
        )
        out = out.join_asof(
            b.select(["open_time", "basis_bps", "basis_z30d"]),
            on="open_time",
            strategy="backward",
        )

    # OI
    if not oi.is_empty():
        o = oi.sort("timestamp").rename({"timestamp": "_oi_ts"}).with_columns(
            pl.col("sum_oi_btc").alias("oi_btc"),
            pl.col("sum_oi_btc").pct_change(BARS_PER_DAY).alias("oi_chg_1d_pct"),
            pl.col("sum_oi_btc").pct_change(16).alias("oi_chg_4h_pct"),
        ).with_columns(
            _rolling_zscore("oi_btc", BARS_PER_DAY * 7, "oi_z7d"),
        )
        out = out.join_asof(
            o.select(["_oi_ts", "oi_btc", "oi_chg_1d_pct", "oi_chg_4h_pct", "oi_z7d"]),
            left_on="open_time",
            right_on="_oi_ts",
            strategy="backward",
        ).drop("_oi_ts")
        # interaction: OI vs price divergence
        out = out.with_columns(
            (
                (pl.col("close").pct_change(16).sign() != pl.col("oi_chg_4h_pct").sign())
            ).cast(pl.Int8).alias("oi_ret_div"),
        )

    # long/short ratio
    if not long_short.is_empty():
        ls = long_short.sort("timestamp").rename({"timestamp": "_ls_ts"}).with_columns(
            pl.col("long_short_ratio").alias("long_short_ratio_top"),
        ).with_columns(
            _rolling_zscore("long_short_ratio_top", BARS_PER_WEEK, "long_short_z7d"),
        )
        out = out.join_asof(
            ls.select(["_ls_ts", "long_short_ratio_top", "long_short_z7d"]),
            left_on="open_time",
            right_on="_ls_ts",
            strategy="backward",
        ).drop("_ls_ts")

    return out


# =========================================================================
# 3. Pipeline orquestrador — plug-in para features.build_v2
# =========================================================================

def add_microstructure_all(
    df: pl.DataFrame,
    basis: pl.DataFrame | None = None,
    oi: pl.DataFrame | None = None,
    long_short: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Aplica toda a stack de microestrutura sobre df já com taker_buy_base.

    Ordem importa: taker_flow → microprice → vpin → basis_oi (joins externos por último).
    """
    out = add_taker_flow(df)
    out = add_microprice(out)
    out = add_vpin(out)
    if basis is not None or oi is not None or long_short is not None:
        out = add_basis_oi(
            out,
            basis if basis is not None else pl.DataFrame(),
            oi if oi is not None else pl.DataFrame(),
            long_short if long_short is not None else pl.DataFrame(),
        )
    return out


# =========================================================================
# 4. Loader stub — wire-up com data/ atual
# =========================================================================

def load_ohlcv_with_taker() -> pl.DataFrame:
    """Carrega ohlcv_15m.parquet. Atualmente NÃO tem taker_buy_base — esse loader
    serve de smoke test e para verificar que precisa do refit do binance.py.

    Quando taker_buy_base existir no parquet, esta função vira o entry point.
    """
    df = pl.read_parquet(DATA / "ohlcv_15m.parquet")
    if "taker_buy_base" not in df.columns:
        print(
            "[WARN] taker_buy_base ausente em ohlcv_15m.parquet. "
            "Refit pipeline/binance.py para preservar campos 9-10 do kline. "
            "Sem isso, add_taker_flow/add_microprice/add_vpin não rodam."
        )
    return df


def load_microstructure_parquets() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Carrega basis/oi/long_short. Retorna DataFrames vazios se ainda não coletados."""
    def _safe_read(name: str) -> pl.DataFrame:
        p = DATA / name
        return pl.read_parquet(p) if p.exists() else pl.DataFrame()

    return _safe_read("basis.parquet"), _safe_read("oi_hist.parquet"), _safe_read("long_short.parquet")


def build_microstructure_matrix() -> pl.DataFrame:
    """Entry point completo: OHLCV + microestrutura, pronto pra concatenar ao build_v2.

    Uso em exp_ensemble.py:
        from proposals import microstructure_features_quant as mq
        df_micro = mq.build_microstructure_matrix()
        # depois fundir com df do build_v2 via open_time join
    """
    ohlcv = load_ohlcv_with_taker()
    basis, oi, long_short = load_microstructure_parquets()
    if "taker_buy_base" not in ohlcv.columns:
        # smoke test — retorna apenas o ohlcv pra que o caller veja o warning
        return ohlcv
    return add_microstructure_all(ohlcv, basis=basis, oi=oi, long_short=long_short)


# =========================================================================
# TODO list — ordem recomendada
# =========================================================================
# [ ] D1. Refit pipeline/binance.py:fetch_klines para preservar campos 9-10
#         (taker_buy_base_asset_volume, taker_buy_quote_asset_volume)
# [ ] D1. Schema migration: ohlcv_15m.parquet + 2 colunas. Rerun ingest_15m.py.
# [ ] D1. Implementar fetch_open_interest_hist, fetch_premium_index (mark/index klines),
#         fetch_top_long_short_position_ratio. Adicionar ao ingest_daily.py ou novo
#         ingest_microstructure.py rodando a cada 15m via GH Action.
# [ ] D2. Wire add_microstructure_all como step novo em features.build_v2 antes do apply_lag.
# [ ] D2. Garantir que apply_lag pega TODAS as novas colunas (LAG_SAFE_EXCLUDE NÃO precisa mudar).
# [ ] D2. Smoke test: nenhuma feature de microestrutura aparece na vela t SEM shift(1).
# [ ] D3. Rerun notebooks/exp_ensemble.py. Ablation por grupo (taker, basis_oi, vpin).
# [ ] D3. SHAP por feature group — confirmar que microestrutura entra no top-15.
# [ ] D4. Update ROADMAP.md Fase 4 e brief de resultados.

if __name__ == "__main__":
    df = build_microstructure_matrix()
    print(f"Shape: {df.shape}")
    print(f"Cols: {df.columns}")
    micro_cols = [c for c in df.columns if any(
        k in c for k in ("taker_buy_ratio", "ofi", "cvd", "microprice", "vpin", "basis", "oi_", "long_short")
    )]
    print(f"Microstructure cols ({len(micro_cols)}): {micro_cols}")
