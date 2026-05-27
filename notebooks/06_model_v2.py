# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # 06 — Model v2: 4h bars + binary + interactions + regime
#
# Mudanças vs v1:
#   - Timeframe: 4h (6 bars/dia, ~7.5k bars em 5y) — menos ruído que 15m
#   - Triple-barrier: ±3×ATR / 12 bars (2 dias)
#   - Binary target: LONG_WIN (1) vs resto (0)
#   - 6 interaction features (funding×ret1d, vix×spx, etc)
#   - 5 regime features (uptrend curto/longo, vol_high, drawdown)
#   - LightGBM binary, threshold sweep
#
# Hipótese: 15m era ruído demais. Macro/sentiment são daily — em 4h sinal entra clean.

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
from sklearn.metrics import log_loss, precision_score, recall_score
from datetime import datetime, timezone
from pipeline import features as feat, labels as lab

TIMEFRAME = 240  # 4h
HORIZON_BARS = 12  # 48h = 2 dias
ATR_MULT = 3.0
COST = 0.0008

# %% [markdown]
# ## 1. Build matrix em 4h

# %%
df = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME, lag=1)
df = df.drop_nulls(subset=["atr_14"])
print(f"shape em 4h: {df.shape}")
print(f"range: {datetime.fromtimestamp(df['open_time'].min()/1000, tz=timezone.utc).date()} → {datetime.fromtimestamp(df['open_time'].max()/1000, tz=timezone.utc).date()}")

# %% [markdown]
# ## 2. Aplica triple-barrier (3×ATR, 2 dias)

# %%
labeled = lab.triple_barrier(df, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=HORIZON_BARS)
print("\nDistribuição labels:")
total = labeled.height
for lbl, name in [(1, "LONG_WIN"), (0, "TIMEOUT"), (-1, "STOP")]:
    n = labeled.filter(pl.col("label") == lbl).height
    rt = labeled.filter(pl.col("label") == lbl)["barrier_ret"].mean()
    print(f"  {name:>8s}  {n:>5d}  ({100*n/total:>4.1f}%)  ret_µ={rt*100 if rt else 0:+.2f}%")

# Binary target: LONG_WIN vs resto
labeled = labeled.with_columns(
    (pl.col("label") == 1).cast(pl.Int8).alias("y_bin")
)
pos_rate = labeled["y_bin"].mean()
print(f"\nBase rate LONG_WIN: {100*pos_rate:.1f}%")

# %% [markdown]
# ## 3. Feature set

# %%
feature_cols = [
    c for c in labeled.columns
    if c not in feat.LAG_SAFE_EXCLUDE
    and c not in {"label", "hit_bar", "barrier_ret", "upper_px", "lower_px", "y_bin"}
]
print(f"Features: {len(feature_cols)}")

# Mantém só linhas com features válidas
mat = labeled.select(["open_time", "close", "y_bin", "barrier_ret", *feature_cols]).drop_nulls(
    subset=feature_cols + ["y_bin"]
).to_pandas()
mat["dt"] = mat["open_time"].apply(lambda ms: datetime.fromtimestamp(ms/1000, tz=timezone.utc))
mat["quarter"] = mat["dt"].dt.to_period("Q")
print(f"Linhas usáveis: {len(mat)}")

# %% [markdown]
# ## 4. Walk-forward binary

# %%
PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l2=0.5,
    is_unbalance=False,  # base rate ~35% próximo de balanceado
    verbose=-1,
    n_jobs=-1,
)
N_ROUNDS = 500

quarters = [q for q in sorted(mat["quarter"].unique()) if q.start_time.year >= 2023]
results = []
all_proba = []
all_y = []
all_ret = []
all_dt = []

import time
for q in quarters:
    test_mask = mat["quarter"] == q
    test_idx = mat.index[test_mask].tolist()
    if not test_idx:
        continue
    test_start = test_idx[0]
    train_end = test_start - HORIZON_BARS
    test_use_start = test_start + HORIZON_BARS
    if train_end < 500 or test_use_start >= test_idx[-1]:
        continue
    train_idx = list(range(0, train_end))
    test_use_idx = [i for i in test_idx if i >= test_use_start]

    X_tr = mat.iloc[train_idx][feature_cols].values
    y_tr = mat.iloc[train_idx]["y_bin"].values
    X_te = mat.iloc[test_use_idx][feature_cols].values
    y_te = mat.iloc[test_use_idx]["y_bin"].values
    ret_te = mat.iloc[test_use_idx]["barrier_ret"].values

    t0 = time.time()
    dtr = lgb.Dataset(X_tr, y_tr)
    model = lgb.train(PARAMS, dtr, num_boost_round=N_ROUNDS)
    dt = time.time() - t0

    proba = model.predict(X_te)
    all_proba.append(proba)
    all_y.append(y_te)
    all_ret.append(ret_te)
    all_dt.extend(mat.iloc[test_use_idx]["dt"].tolist())

    # Default threshold 0.5
    pred = (proba > 0.5).astype(int)
    take = pred == 1
    n_sig = take.sum()
    avg_pnl = (ret_te[take] - COST).mean() if n_sig > 0 else 0
    tot_pnl = (ret_te[take] - COST).sum() if n_sig > 0 else 0
    win_rate = ((ret_te[take] - COST) > 0).mean() if n_sig > 0 else 0
    base = y_te.mean()
    prec = precision_score(y_te, pred, zero_division=0)

    results.append({
        "quarter": str(q),
        "n_train": len(train_idx),
        "n_test": len(test_use_idx),
        "base_rate": base,
        "n_signals": int(n_sig),
        "precision": prec,
        "avg_pnl": avg_pnl,
        "tot_pnl": tot_pnl,
        "win_rate": win_rate,
        "secs": dt,
    })
    print(
        f"{str(q):>8s}  tr={len(train_idx):>5d}  te={len(test_use_idx):>4d}  "
        f"base={100*base:>4.1f}%  sinais={int(n_sig):>4d}  prec={100*prec:>4.1f}%  "
        f"win={100*win_rate:>4.1f}%  avgPnL={100*avg_pnl:+.3f}%  totPnL={100*tot_pnl:+.2f}%  ({dt:.1f}s)"
    )

last_model = model

# %% [markdown]
# ## 5. Sumário + threshold sweep agregado

# %%
proba_all = np.concatenate(all_proba)
y_all = np.concatenate(all_y)
ret_all = np.concatenate(all_ret)

print(f"\n=== Pool agregado: {len(y_all)} amostras, base rate {100*y_all.mean():.1f}% ===")
print(f"{'threshold':>10s}  {'n_sig':>6s}  {'rate':>5s}  {'prec':>5s}  {'win':>5s}  {'avg':>8s}  {'tot':>9s}")
for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    take = proba_all > thr
    n = take.sum()
    if n < 10:
        continue
    pnl = ret_all[take] - COST
    win = (pnl > 0).mean()
    avg = pnl.mean()
    tot = pnl.sum()
    pred = take.astype(int)
    prec = precision_score(y_all, pred, zero_division=0)
    print(f"  >{thr:.2f}      {n:>5d}  {100*n/len(y_all):>4.1f}%  {100*prec:>4.1f}%  {100*win:>4.1f}%  {100*avg:+.3f}%  {100*tot:+7.2f}%")

# %% [markdown]
# ## 6. Feature importance (último fold)

# %%
imp = pd.DataFrame({
    "feature": feature_cols,
    "gain": last_model.feature_importance(importance_type="gain"),
    "split": last_model.feature_importance(importance_type="split"),
}).sort_values("gain", ascending=False)
print("\nTop 20 features por GAIN:")
print(imp.head(20).to_string(index=False))

# %% [markdown]
# ## 7. Sumário final

# %%
res = pd.DataFrame(results)
print(f"\nFolds: {len(res)}  |  positivos: {(res['tot_pnl']>0).sum()}/{len(res)} ({100*(res['tot_pnl']>0).mean():.0f}%)")
print(f"PnL acumulado total:      {100*res['tot_pnl'].sum():+.2f}%")
print(f"Win rate médio:           {100*res['win_rate'].mean():.1f}%")
print(f"Precision média:          {100*res['precision'].mean():.1f}%")
print(f"Avg PnL/sinal médio:      {100*res['avg_pnl'].mean():+.3f}%")
print(f"Sinais/trim médio:        {res['n_signals'].mean():.0f}")
