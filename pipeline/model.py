"""Treino + predição reutilizável do modelo v2 (LightGBM binary, 4h bars).

Params fixos validados no notebook 06. Não fazemos hyperopt em produção — overfita.
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from pipeline import features as feat, labels as lab

TIMEFRAME_MIN = 240
HORIZON_BARS = 12          # mid horizon (48h)
HORIZON_BARS_LONG = 18     # long horizon (72h) — segundo modelo do AND
ATR_MULT = 3.0
SIGNAL_THRESHOLD = 0.35
RISK_PER_TRADE = 0.01      # 1% do capital arriscado no stop (validado em exp_position_sizing)
NO_BEAR_THRESHOLD = -0.05  # se BTC caiu >5% no último mês → suprime sinal (validado em exp_regime_analysis)

LGB_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l2=0.5,
    verbose=-1,
    n_jobs=-1,
)
N_ROUNDS = 500


def build_training_matrix(horizon_bars: int = HORIZON_BARS) -> tuple[pl.DataFrame, list[str]]:
    """Constrói matriz completa + lista de feature columns para um horizonte específico."""
    df = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME_MIN, lag=1).drop_nulls(subset=["atr_14"])
    labeled = lab.triple_barrier(df, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=horizon_bars)
    labeled = labeled.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y"))
    feature_cols = [
        c for c in labeled.columns
        if c not in feat.LAG_SAFE_EXCLUDE
        and c not in {"label", "hit_bar", "barrier_ret", "upper_px", "lower_px", "y"}
    ]
    mat = labeled.select(["open_time", "close", "y", "barrier_ret", *feature_cols])
    mat = mat.drop_nulls(subset=feature_cols + ["y"])
    return mat, feature_cols


def train(mat: pl.DataFrame, feature_cols: list[str], horizon_bars: int = HORIZON_BARS) -> lgb.Booster:
    """Treina com purge: exclui últimas `horizon_bars` linhas do treino
    (cujos labels dependem de futuro que ainda não temos).
    """
    use = mat.head(mat.height - horizon_bars)
    X = use.select(feature_cols).to_numpy()
    y = use["y"].to_numpy()
    return lgb.train(LGB_PARAMS, lgb.Dataset(X, y), num_boost_round=N_ROUNDS)


def predict_latest(model: lgb.Booster, mat: pl.DataFrame, feature_cols: list[str]) -> dict:
    """Prediz na vela mais recente (single model — usado pra debug/baseline)."""
    last = mat.tail(1)
    X = last.select(feature_cols).to_numpy()
    proba = float(model.predict(X)[0])
    return {
        "open_time": int(last["open_time"][0]),
        "close": float(last["close"][0]),
        "proba_long": proba,
        "signal": proba > SIGNAL_THRESHOLD,
        "confidence_pct": (proba - SIGNAL_THRESHOLD) / (1 - SIGNAL_THRESHOLD) * 100,
    }


def predict_dual_horizon() -> dict:
    """Pipeline completa do modelo de PRODUÇÃO (dual-horizon AND).

    Treina 2 modelos (mid=12 bars=48h, long=18 bars=72h) e prediz na vela
    mais recente comum aos dois. Sinal = AMBOS > threshold.

    Validado no exp `exp_multi_horizon` + teste manual: Sharpe 1.29 vs 0.88 do mid sozinho.
    """
    mat_mid, fc_mid = build_training_matrix(horizon_bars=HORIZON_BARS)
    mat_long, fc_long = build_training_matrix(horizon_bars=HORIZON_BARS_LONG)

    m_mid = train(mat_mid, fc_mid, horizon_bars=HORIZON_BARS)
    m_long = train(mat_long, fc_long, horizon_bars=HORIZON_BARS_LONG)

    last_mid = mat_mid.tail(1)
    proba_mid = float(m_mid.predict(last_mid.select(fc_mid).to_numpy())[0])

    # Match última vela do long pela mesma open_time
    ot = int(last_mid["open_time"][0])
    long_row = mat_long.filter(pl.col("open_time") == ot)
    if long_row.is_empty():
        # Long matrix pode ser ligeiramente mais curta (purge maior) — usa a última disponível
        long_row = mat_long.tail(1)
    proba_long_h = float(m_long.predict(long_row.select(fc_long).to_numpy())[0])

    # Filtro de regime: suprime sinal se BTC caiu mais que NO_BEAR_THRESHOLD no último mês.
    # Mês = ~180 bars 4h (30 dias × 6 bars/dia). Compara close atual vs close ~30d atrás.
    bars_per_month = 180
    if mat_mid.height >= bars_per_month + 1:
        close_now = float(mat_mid["close"][-1])
        close_30d_ago = float(mat_mid["close"][-1 - bars_per_month])
        ret_30d = close_now / close_30d_ago - 1
    else:
        ret_30d = 0.0  # warm-up insuficiente, deixa passar

    in_bear = ret_30d < NO_BEAR_THRESHOLD

    # Sinal final: AMBOS concordam E não está em bear
    signal_mid = proba_mid > SIGNAL_THRESHOLD
    signal_long_h = proba_long_h > SIGNAL_THRESHOLD
    signal_ml = signal_mid and signal_long_h
    signal = signal_ml and not in_bear

    return {
        "open_time": ot,
        "close": float(last_mid["close"][0]),
        "proba_mid": proba_mid,
        "proba_long_horizon": proba_long_h,
        "proba_long": proba_mid,   # backward compat com format_signal
        "signal_mid": signal_mid,
        "signal_long_h": signal_long_h,
        "signal_ml": signal_ml,
        "in_bear": in_bear,
        "ret_30d": ret_30d,
        "signal": signal,
        "confidence_pct": (min(proba_mid, proba_long_h) - SIGNAL_THRESHOLD) / (1 - SIGNAL_THRESHOLD) * 100,
        "_mat_mid": mat_mid,
        "_features_mid": fc_mid,
    }


def position_size(
    capital: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float = RISK_PER_TRADE,
    max_pct: float = 0.50,
) -> dict:
    """Calcula tamanho de posição risk-based (1% do capital no stop).

    Retorna {size_btc, size_usd, risk_dollars, pct_of_capital, capped}.
    Cap em max_pct (default 50%) pra evitar que ATR baixo gere posição enorme.
    """
    risk_dollars = capital * risk_pct
    distance = entry_price - stop_price
    if distance <= 0:
        return {"size_btc": 0, "size_usd": 0, "risk_dollars": risk_dollars, "pct_of_capital": 0, "capped": False}
    size_btc = risk_dollars / distance
    size_usd = size_btc * entry_price
    pct = size_usd / capital
    capped = pct > max_pct
    if capped:
        size_usd = capital * max_pct
        size_btc = size_usd / entry_price
        pct = max_pct
    return {
        "size_btc": size_btc,
        "size_usd": size_usd,
        "risk_dollars": risk_dollars,
        "pct_of_capital": pct,
        "capped": capped,
    }
