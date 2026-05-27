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
SIZING_MODE = "full"       # "full" = 100% do capital, "risk1" = 1% risk on stop
RISK_PER_TRADE = 0.01      # usado se SIZING_MODE="risk1" — sizing conservador
NO_BEAR_THRESHOLD = -0.05  # se BTC caiu >5% no último mês → suprime sinal (validado em exp_regime_analysis)
ENSEMBLE_RULE = "MID"      # winner A1-A (Red Team M5 passou): MID sozinho > AND.
                           # Valores: "MID" (prod) | "AND" (legado dual-horizon) | "OR"

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
    """Constrói matriz completa + lista de feature columns para um horizonte específico.

    Inclui `uniqueness_weight` (LdP eq.4.2) pra ponderar samples sobrepostos no LightGBM.
    """
    df = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME_MIN, lag=1).drop_nulls(subset=["atr_14"])
    labeled = lab.triple_barrier(df, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=horizon_bars)
    labeled = lab.attach_uniqueness(labeled, horizon_bars=horizon_bars)
    labeled = labeled.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y"))
    feature_cols = [
        c for c in labeled.columns
        if c not in feat.LAG_SAFE_EXCLUDE
        and c not in {"label", "hit_bar", "barrier_ret", "upper_px", "lower_px", "y", "uniqueness_weight"}
    ]
    mat = labeled.select(["open_time", "close", "y", "barrier_ret", "uniqueness_weight", *feature_cols])
    mat = mat.drop_nulls(subset=feature_cols + ["y"])
    return mat, feature_cols


def train(mat: pl.DataFrame, feature_cols: list[str], horizon_bars: int = HORIZON_BARS) -> lgb.Booster:
    """Treina com purge: exclui últimas `horizon_bars` linhas do treino
    (cujos labels dependem de futuro que ainda não temos).

    Aplica `uniqueness_weight` (LdP) em `lgb.Dataset(weight=...)` se disponível.
    """
    use = mat.head(mat.height - horizon_bars)
    X = use.select(feature_cols).to_numpy()
    y = use["y"].to_numpy()
    weight = use["uniqueness_weight"].to_numpy() if "uniqueness_weight" in use.columns else None
    return lgb.train(LGB_PARAMS, lgb.Dataset(X, y, weight=weight), num_boost_round=N_ROUNDS)


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
    """Pipeline completa do modelo de PRODUÇÃO.

    Mantém o nome "dual_horizon" por compat com workflows, mas a REGRA atual
    (ENSEMBLE_RULE) é configurável. Default "MID" (winner A1-A validado por
    Red Team M5): apenas o modelo mid decide o sinal. Long fica informativo
    no payload pra debug.

    Regras:
      MID — só mid (winner: VAL Sharpe 0.36 / HOLDOUT 1.53 / PSR 0.952)
      AND — mid AND long (legado dual-horizon, pior em backtest honesto)
      OR  — mid OR long (alta cobertura, geralmente Sharpe pior)
    """
    mat_mid, fc_mid = build_training_matrix(horizon_bars=HORIZON_BARS)
    m_mid = train(mat_mid, fc_mid, horizon_bars=HORIZON_BARS)

    last_mid = mat_mid.tail(1)
    proba_mid = float(m_mid.predict(last_mid.select(fc_mid).to_numpy())[0])
    ot = int(last_mid["open_time"][0])

    # Long-horizon: só treina/prediz se regra precisar (economiza ~30s no cron)
    proba_long_h = None
    if ENSEMBLE_RULE in ("AND", "OR"):
        mat_long, fc_long = build_training_matrix(horizon_bars=HORIZON_BARS_LONG)
        m_long = train(mat_long, fc_long, horizon_bars=HORIZON_BARS_LONG)
        long_row = mat_long.filter(pl.col("open_time") == ot)
        if long_row.is_empty():
            long_row = mat_long.tail(1)
        proba_long_h = float(m_long.predict(long_row.select(fc_long).to_numpy())[0])

    # Filtro de regime: suprime sinal se BTC caiu mais que NO_BEAR_THRESHOLD no último mês.
    bars_per_month = 180
    if mat_mid.height >= bars_per_month + 1:
        close_now = float(mat_mid["close"][-1])
        close_30d_ago = float(mat_mid["close"][-1 - bars_per_month])
        ret_30d = close_now / close_30d_ago - 1
    else:
        ret_30d = 0.0  # warm-up insuficiente, deixa passar

    in_bear = ret_30d < NO_BEAR_THRESHOLD

    # Sinal por regra
    signal_mid = proba_mid > SIGNAL_THRESHOLD
    signal_long_h = (proba_long_h is not None) and (proba_long_h > SIGNAL_THRESHOLD)
    if ENSEMBLE_RULE == "MID":
        signal_ml = signal_mid
    elif ENSEMBLE_RULE == "AND":
        signal_ml = signal_mid and signal_long_h
    elif ENSEMBLE_RULE == "OR":
        signal_ml = signal_mid or signal_long_h
    else:
        raise ValueError(f"ENSEMBLE_RULE desconhecido: {ENSEMBLE_RULE}")

    signal = signal_ml and not in_bear

    # Confidence baseado em proba_mid (regra MID em prod)
    confidence_pct = (proba_mid - SIGNAL_THRESHOLD) / (1 - SIGNAL_THRESHOLD) * 100

    return {
        "open_time": ot,
        "close": float(last_mid["close"][0]),
        "proba_mid": proba_mid,
        "proba_long_horizon": proba_long_h,  # None se regra=MID
        "proba_long": proba_mid,   # backward compat com format_signal
        "signal_mid": signal_mid,
        "signal_long_h": signal_long_h,
        "signal_ml": signal_ml,
        "ensemble_rule": ENSEMBLE_RULE,
        "in_bear": in_bear,
        "ret_30d": ret_30d,
        "signal": signal,
        "confidence_pct": confidence_pct,
        "_mat_mid": mat_mid,
        "_features_mid": fc_mid,
    }


def position_size(
    capital: float,
    entry_price: float,
    stop_price: float,
    mode: str | None = None,
    risk_pct: float = RISK_PER_TRADE,
    max_pct: float = 0.50,
) -> dict:
    """Calcula tamanho de posição.

    Mode "full" (default em produção): 100% do capital.
    Mode "risk1": 1% do capital arriscado no stop (mais conservador).

    Retorna {size_btc, size_usd, risk_dollars, pct_of_capital, capped, mode}.
    """
    mode = mode or SIZING_MODE
    if mode == "full":
        size_usd = capital
        size_btc = capital / entry_price
        # risco real = (entry - stop) / entry × size_usd
        risk_dollars = (entry_price - stop_price) / entry_price * size_usd
        return {
            "size_btc": size_btc,
            "size_usd": size_usd,
            "risk_dollars": risk_dollars,
            "pct_of_capital": 1.0,
            "capped": False,
            "mode": "full",
        }

    # mode == "risk1"
    risk_dollars = capital * risk_pct
    distance = entry_price - stop_price
    if distance <= 0:
        return {"size_btc": 0, "size_usd": 0, "risk_dollars": risk_dollars,
                "pct_of_capital": 0, "capped": False, "mode": "risk1"}
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
        "mode": "risk1",
    }
