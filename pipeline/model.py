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

    # Sinal final: AMBOS concordam
    signal_mid = proba_mid > SIGNAL_THRESHOLD
    signal_long_h = proba_long_h > SIGNAL_THRESHOLD
    signal = signal_mid and signal_long_h

    return {
        "open_time": ot,
        "close": float(last_mid["close"][0]),
        "proba_mid": proba_mid,
        "proba_long_horizon": proba_long_h,
        "proba_long": proba_mid,   # backward compat com format_signal
        "signal_mid": signal_mid,
        "signal_long_h": signal_long_h,
        "signal": signal,
        "confidence_pct": (min(proba_mid, proba_long_h) - SIGNAL_THRESHOLD) / (1 - SIGNAL_THRESHOLD) * 100,
        "_mat_mid": mat_mid,
        "_features_mid": fc_mid,
    }
