# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # 09 — Meta-labeling (López de Prado cap. 3.6)
#
# Modelo PRIMÁRIO: v2 atual, prediz LONG_WIN. Saída: proba ∈ [0,1].
# Modelo META: dado um sinal do primário (proba > thr_lower), decide se confiar.
#   Treina: features + proba_primary → label "primário acertou?"
#   Output: P(acertar | sinalizei)
#
# Sinal final = primário sinaliza  AND  meta concorda (proba_meta > 0.5)
#
# Hipótese: meta filtra falsos positivos do primário → precision sobe,
# Sharpe melhora. Coverage cai (menos sinais), mas qualidade aumenta.

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

# %% [markdown]
# ## 1. Build matriz v2

# %%
df = feat.build_v2_from_parquets(timeframe_min=240, lag=1).drop_nulls(subset=["atr_14"])
labeled = lab.triple_barrier(df, upper_mult=3.0, lower_mult=3.0, horizon_bars=12)
labeled = labeled.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y"))
feature_cols = [c for c in labeled.columns if c not in feat.LAG_SAFE_EXCLUDE and c not in {"label","hit_bar","barrier_ret","upper_px","lower_px","y"}]
mat = labeled.select(["open_time","close","y","barrier_ret", *feature_cols]).drop_nulls(subset=feature_cols+["y"]).to_pandas()
mat["dt"] = mat["open_time"].apply(lambda ms: datetime.fromtimestamp(ms/1000, tz=timezone.utc))
mat["quarter"] = mat["dt"].dt.to_period("Q")
print(f"rows={len(mat)}  features={len(feature_cols)}")

PARAMS_PRIM = dict(objective="binary", metric="binary_logloss", verbose=-1, n_jobs=-1,
                   learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, lambda_l2=0.5)
PARAMS_META = dict(objective="binary", metric="binary_logloss", verbose=-1, n_jobs=-1,
                   learning_rate=0.05, num_leaves=15, min_data_in_leaf=50,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0)
COST = 0.0015  # Binance taker 0.10% × 2 + slippage real
HORIZON = 12
THR_PRIM = 0.30  # lower threshold pra primário (deixa meta filtrar)
THR_META = 0.50


# %% [markdown]
# ## 2. Walk-forward com 2 níveis
#
# Para cada fold:
#   train_idx: histórico antes de fold_start - HORIZON  (purge)
#   primário treina, prediz no histórico mesmo (in-sample) — só pra gerar dataset META
#   meta treina onde primário sinalizou (proba > THR_PRIM)
#   ambos preveem no test fold

# %%
quarters = [q for q in sorted(mat["quarter"].unique()) if q.start_time.year >= 2023]
results = []

for q in quarters:
    test_idx = mat.index[mat["quarter"] == q].tolist()
    if not test_idx: continue
    train_end = test_idx[0] - HORIZON
    test_use_start = test_idx[0] + HORIZON
    if train_end < 500 or test_use_start >= test_idx[-1]: continue
    train_idx = list(range(0, train_end))
    test_use_idx = [i for i in test_idx if i >= test_use_start]

    # === PRIMARY ===
    X_tr = mat.iloc[train_idx][feature_cols].values
    y_tr = mat.iloc[train_idx]["y"].values
    model_prim = lgb.train(PARAMS_PRIM, lgb.Dataset(X_tr, y_tr), num_boost_round=500)

    # OOB predictions no próprio train via 2-fold pra dataset meta
    # (evitar leak: predito por modelo treinado SEM aquela barra)
    # Approximação simples: re-treina em primeira metade, prediz segunda; e vice-versa
    half = len(train_idx) // 2
    fold_a = train_idx[:half]
    fold_b = train_idx[half:]
    Xa = mat.iloc[fold_a][feature_cols].values
    ya = mat.iloc[fold_a]["y"].values
    Xb = mat.iloc[fold_b][feature_cols].values
    yb = mat.iloc[fold_b]["y"].values
    m_a = lgb.train(PARAMS_PRIM, lgb.Dataset(Xa, ya), num_boost_round=500)
    m_b = lgb.train(PARAMS_PRIM, lgb.Dataset(Xb, yb), num_boost_round=500)
    pa = m_b.predict(Xa)  # fold_a predito por modelo SEM fold_a
    pb = m_a.predict(Xb)
    oob_proba = np.concatenate([pa, pb])

    # === META ===
    # Pega só bars onde primário sinalizou
    primary_signaled = oob_proba > THR_PRIM
    meta_X = mat.iloc[train_idx][feature_cols].values[primary_signaled]
    # adiciona proba_primary como feature meta
    meta_X = np.column_stack([meta_X, oob_proba[primary_signaled]])
    # Label meta = primário acertou? (1 se y_real=1, 0 caso contrário)
    meta_y = mat.iloc[train_idx]["y"].values[primary_signaled]

    if len(meta_y) < 100 or len(set(meta_y)) < 2:
        print(f"  {str(q):>8s}  meta dataset insuficiente ({len(meta_y)} samples)")
        continue

    model_meta = lgb.train(PARAMS_META, lgb.Dataset(meta_X, meta_y), num_boost_round=300)

    # === TEST ===
    X_te = mat.iloc[test_use_idx][feature_cols].values
    proba_prim_te = model_prim.predict(X_te)
    ret_te = mat.iloc[test_use_idx]["barrier_ret"].values

    # Stage 1: primário sinaliza?
    stage1 = proba_prim_te > THR_PRIM
    # Stage 2: meta concorda?
    meta_X_te = np.column_stack([X_te[stage1], proba_prim_te[stage1]])
    if len(meta_X_te) > 0:
        proba_meta_te = model_meta.predict(meta_X_te)
        stage2 = proba_meta_te > THR_META
    else:
        stage2 = np.array([], dtype=bool)

    # Sinal final = primário AND meta
    take = np.zeros(len(test_use_idx), dtype=bool)
    take[stage1] = stage2

    # Baseline: só primário (sem meta) com mesmo threshold
    baseline = stage1

    # Métricas
    def metrics(take_arr, label):
        n = int(take_arr.sum())
        if n < 5:
            return dict(label=label, n=n, total=0, win=0, sharpe=0)
        pnl = ret_te[take_arr] - COST
        total = pnl.sum()
        win = (pnl > 0).mean()
        strat = np.where(take_arr, ret_te - COST, 0.0)
        sharpe = (strat.mean()/strat.std())*np.sqrt(6*365) if strat.std()>0 else 0
        return dict(label=label, n=n, total=total, win=win, sharpe=sharpe)

    m_baseline = metrics(baseline, "primary-only")
    m_meta = metrics(take, "primary+meta")
    print(
        f"  {str(q):>8s}  "
        f"prim={m_baseline['n']:>4d}/PnL{100*m_baseline['total']:+6.1f}%/win{100*m_baseline['win']:.0f}%  "
        f"meta={m_meta['n']:>4d}/PnL{100*m_meta['total']:+6.1f}%/win{100*m_meta['win']:.0f}%"
    )
    results.append((str(q), m_baseline, m_meta))


# %% [markdown]
# ## 3. Agregado

# %%
import pandas as pd
rows_b = pd.DataFrame([r[1] for r in results])
rows_m = pd.DataFrame([r[2] for r in results])
print("\n=== Agregado walk-forward 2023Q1 → 2026Q2 ===")
print(f"{'modelo':<18s}  {'sinais':>7s}  {'PnL total':>10s}  {'win%':>5s}  {'sharpe_µ':>9s}  {'+folds':>6s}")
print(f"  {'primary-only':<16s}  {rows_b['n'].sum():>7d}  {100*rows_b['total'].sum():>+9.1f}%  {100*rows_b['win'].mean():>4.1f}%  {rows_b['sharpe'].mean():>+8.2f}  {(rows_b['total']>0).sum():>3d}/{len(rows_b)}")
print(f"  {'primary+meta':<16s}  {rows_m['n'].sum():>7d}  {100*rows_m['total'].sum():>+9.1f}%  {100*rows_m['win'].mean():>4.1f}%  {rows_m['sharpe'].mean():>+8.2f}  {(rows_m['total']>0).sum():>3d}/{len(rows_m)}")

# %% [markdown]
# ## 4. Comparação concentrada nos folds maduros (2025+)

# %%
mature_b = rows_b.tail(6) if len(rows_b) >= 6 else rows_b
mature_m = rows_m.tail(6) if len(rows_m) >= 6 else rows_m
print(f"\n=== Folds maduros (últimos 6 trimestres) ===")
print(f"  primary-only:  PnL {100*mature_b['total'].sum():+.1f}%  win {100*mature_b['win'].mean():.1f}%  sharpe_µ {mature_b['sharpe'].mean():+.2f}")
print(f"  primary+meta:  PnL {100*mature_m['total'].sum():+.1f}%  win {100*mature_m['win'].mean():.1f}%  sharpe_µ {mature_m['sharpe'].mean():+.2f}")
