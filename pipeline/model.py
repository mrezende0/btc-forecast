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
HORIZON_BARS = 12
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


def build_training_matrix() -> tuple[pl.DataFrame, list[str]]:
    """Constrói matriz completa + lista de feature columns."""
    df = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME_MIN, lag=1).drop_nulls(subset=["atr_14"])
    labeled = lab.triple_barrier(df, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=HORIZON_BARS)
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
    """Prediz na vela mais recente (a que ainda não tem label completo).

    Retorna dict com: timestamp, close, proba_long, signal (bool), confidence_pct.
    """
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
