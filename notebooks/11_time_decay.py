# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # 11 — Time-decay weighting
#
# Hipótese: BTC regime mudou ao longo do tempo (2021 retail, 2022 bear, 2023+
# institucional/ETF). Samples antigos podem ter pattern obsoleto. Pesar mais
# os recentes pode melhorar generalização no presente.
#
# Implementação: LGB sample_weight = exp(-(N-i)/tau) onde N=tamanho train, i=índice.
# Tau controla agressividade do decay. tau=∞ = sem decay (v2 atual).
# tau=N/2 = sample mais antigo pesa ~37% do mais recente.

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
from datetime import datetime, timezone
from pipeline import features as feat, labels as lab

# %%
df = feat.build_v2_from_parquets(timeframe_min=240, lag=1).drop_nulls(subset=["atr_14"])
labeled = lab.triple_barrier(df, upper_mult=3.0, lower_mult=3.0, horizon_bars=12)
labeled = labeled.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y"))
feature_cols = [c for c in labeled.columns if c not in feat.LAG_SAFE_EXCLUDE and c not in {"label","hit_bar","barrier_ret","upper_px","lower_px","y"}]
mat = labeled.select(["open_time","close","y","barrier_ret", *feature_cols]).drop_nulls(subset=feature_cols+["y"]).to_pandas()
mat["dt"] = mat["open_time"].apply(lambda ms: datetime.fromtimestamp(ms/1000, tz=timezone.utc))
mat["quarter"] = mat["dt"].dt.to_period("Q")

PARAMS = dict(objective="binary", metric="binary_logloss", verbose=-1, n_jobs=-1,
              learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, lambda_l2=0.5)
COST = 0.0015  # Binance taker 0.10% × 2 + slippage real
HORIZON = 12
THR = 0.35

quarters = [q for q in sorted(mat["quarter"].unique()) if q.start_time.year >= 2023]


# %%
def run(tau_frac: float | None = None) -> dict:
    """tau_frac = fração do train_size pra tau. None = sem decay (v2 baseline)."""
    all_proba, all_ret = [], []
    for q in quarters:
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
        ret_te = mat.iloc[test_use_idx]["barrier_ret"].values

        if tau_frac is None:
            ds = lgb.Dataset(X_tr, y_tr)
        else:
            N = len(train_idx)
            tau = N * tau_frac
            i = np.arange(N)
            w = np.exp(-(N - 1 - i) / tau)  # peso 1.0 no mais recente, decaindo
            ds = lgb.Dataset(X_tr, y_tr, weight=w)
        model = lgb.train(PARAMS, ds, num_boost_round=500)
        all_proba.append(model.predict(X_te))
        all_ret.append(ret_te)
    proba = np.concatenate(all_proba)
    ret = np.concatenate(all_ret)
    take = proba > THR
    n = int(take.sum())
    strat = np.where(take, ret - COST, 0.0)
    total = np.cumprod(1 + strat)[-1] - 1
    sharpe = (strat.mean()/strat.std()) * np.sqrt(6*365) if strat.std() > 0 else 0
    pnl_nz = strat[take]
    win = (pnl_nz > 0).mean() if n else 0
    pf = pnl_nz[pnl_nz>0].sum() / max(1e-9, -pnl_nz[pnl_nz<0].sum())
    dd = (np.cumprod(1+strat) / np.maximum.accumulate(np.cumprod(1+strat)) - 1).min()
    return dict(n=n, total=total, sharpe=sharpe, win=win, pf=pf, dd=dd)


# %% [markdown]
# ## Sweep de tau

# %%
print(f"{'tau_frac':>10s}  {'n':>5s}  {'total':>8s}  {'CAGR':>7s}  {'Sharpe':>7s}  {'win%':>5s}  {'PF':>5s}  {'MaxDD':>7s}")
configs = [
    ("no decay (v2)", None),
    ("tau=2.0×N", 2.0),
    ("tau=1.0×N", 1.0),
    ("tau=0.5×N", 0.5),
    ("tau=0.33×N", 0.33),
    ("tau=0.25×N", 0.25),
    ("tau=0.15×N", 0.15),
]
for label, tau in configs:
    r = run(tau)
    cagr = (1 + r['total']) ** (1/3.5) - 1
    print(f"  {label:<14s}  {r['n']:>5d}  {100*r['total']:>+7.1f}%  {100*cagr:>+6.1f}%  {r['sharpe']:>+6.2f}  {100*r['win']:>4.1f}%  {r['pf']:>5.2f}  {100*r['dd']:>+6.1f}%")
