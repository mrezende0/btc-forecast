# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # 07 — Hyperopt LightGBM (Optuna)
#
# Otimiza params do v2 maximizando Sharpe walk-forward.
# **Importante:** hyperopt corre só DENTRO de uma janela de validação (purged),
# nunca olhando o teste final. Senão contamina.
#
# Estratégia:
#   - Train: dados <= 2024Q4 - HORIZON
#   - Val:   2024Q1 → 2024Q4 (4 folds purged)
#   - Holdout (não tocado): 2025Q1 → 2026Q2
#   - Otimiza Sharpe em VAL, reporta em HOLDOUT

# %%
from __future__ import annotations
from pathlib import Path
import sys, os
ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from datetime import datetime, timezone
from pipeline import features as feat, labels as lab
optuna.logging.set_verbosity(optuna.logging.WARNING)

TIMEFRAME = 240
HORIZON = 12
ATR_MULT = 3.0
COST = 0.0015  # Binance taker 0.10% × 2 + slippage real
N_TRIALS = 30
THRESHOLD = 0.35

# %%
df = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME, lag=1).drop_nulls(subset=["atr_14"])
labeled = lab.triple_barrier(df, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=HORIZON)
labeled = labeled.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y"))
feature_cols = [c for c in labeled.columns if c not in feat.LAG_SAFE_EXCLUDE and c not in {"label","hit_bar","barrier_ret","upper_px","lower_px","y"}]
mat = labeled.select(["open_time","close","y","barrier_ret", *feature_cols]).drop_nulls(subset=feature_cols+["y"]).to_pandas()
mat["dt"] = mat["open_time"].apply(lambda ms: datetime.fromtimestamp(ms/1000, tz=timezone.utc))
mat["quarter"] = mat["dt"].dt.to_period("Q")

VAL_QUARTERS = [q for q in sorted(mat["quarter"].unique()) if 2024 == q.start_time.year]
HOLDOUT_QUARTERS = [q for q in sorted(mat["quarter"].unique()) if q.start_time.year >= 2025]
print(f"VAL: {VAL_QUARTERS}")
print(f"HOLDOUT: {HOLDOUT_QUARTERS}")


# %%
def run_walk_forward(params: dict, test_quarters, n_rounds: int) -> dict:
    """Roda walk-forward expanding nos quarters dados. Retorna métricas agregadas."""
    all_proba, all_ret = [], []
    for q in test_quarters:
        test_idx = mat.index[mat["quarter"] == q].tolist()
        if not test_idx: continue
        train_end = test_idx[0] - HORIZON
        test_use_start = test_idx[0] + HORIZON
        if train_end < 500 or test_use_start >= test_idx[-1]: continue
        train_idx = list(range(0, train_end))
        test_use_idx = [i for i in test_idx if i >= test_use_start]
        X_tr = mat.iloc[train_idx][feature_cols].values
        y_tr = mat.iloc[train_idx]["y"].values
        X_te = mat.iloc[test_use_idx][feature_cols].values
        model = lgb.train(params, lgb.Dataset(X_tr, y_tr), num_boost_round=n_rounds)
        all_proba.append(model.predict(X_te))
        all_ret.append(mat.iloc[test_use_idx]["barrier_ret"].values)
    if not all_proba:
        return {"sharpe": -99}
    proba = np.concatenate(all_proba)
    ret = np.concatenate(all_ret)
    take = proba > THRESHOLD
    if take.sum() < 10:
        return {"sharpe": -99}
    strat = np.where(take, ret - COST, 0.0)
    sharpe = (strat.mean() / strat.std()) * np.sqrt(6*365) if strat.std() > 0 else 0
    eq = np.cumprod(1 + strat)
    total = eq[-1] - 1
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return {"sharpe": sharpe, "total": total, "dd": dd, "n_sig": int(take.sum())}


def objective(trial: optuna.Trial) -> float:
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbose": -1,
        "n_jobs": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 30, 300),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 2.0),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.5),
    }
    n_rounds = trial.suggest_int("n_rounds", 200, 800, step=100)
    result = run_walk_forward(params, VAL_QUARTERS, n_rounds)
    return result["sharpe"]


# %% [markdown]
# ## Roda hyperopt na VAL

# %%
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f"\nBest VAL Sharpe: {study.best_value:.3f}")
print(f"Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## Avalia no HOLDOUT

# %%
best_params = {
    "objective": "binary", "metric": "binary_logloss", "verbose": -1, "n_jobs": -1,
    **{k: v for k, v in study.best_params.items() if k != "n_rounds"},
}
n_rounds = study.best_params["n_rounds"]
holdout_result = run_walk_forward(best_params, HOLDOUT_QUARTERS, n_rounds)
print(f"\n=== HOLDOUT (2025Q1 → 2026Q2) ===")
print(f"  Sharpe:        {holdout_result['sharpe']:+.3f}")
print(f"  Total return:  {100*holdout_result['total']:+.2f}%")
print(f"  MaxDD:         {100*holdout_result['dd']:+.2f}%")
print(f"  N sinais:      {holdout_result['n_sig']}")

# %% [markdown]
# ## Baseline (params atuais) — comparação justa no mesmo holdout

# %%
baseline_params = dict(
    objective="binary", metric="binary_logloss", verbose=-1, n_jobs=-1,
    learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, lambda_l2=0.5,
)
baseline_holdout = run_walk_forward(baseline_params, HOLDOUT_QUARTERS, 500)
print(f"\n=== BASELINE HOLDOUT (mesmos quarters) ===")
print(f"  Sharpe:        {baseline_holdout['sharpe']:+.3f}")
print(f"  Total:         {100*baseline_holdout['total']:+.2f}%")
print(f"  MaxDD:         {100*baseline_holdout['dd']:+.2f}%")
print(f"  N sinais:      {baseline_holdout['n_sig']}")

delta_sharpe = holdout_result["sharpe"] - baseline_holdout["sharpe"]
print(f"\nΔ Sharpe (hyperopt vs baseline): {delta_sharpe:+.3f}")
