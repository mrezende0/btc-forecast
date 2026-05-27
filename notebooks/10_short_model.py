# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # 10 — Modelo SHORT espelhado
#
# v2 atual prediz LONG_WIN (label=+1). Aqui treinamos modelo simétrico que
# prediz STOP (label=-1) → sinal de venda/short.
#
# Hipótese: se modelo tem edge em direção positiva, deve ter no espelho.
# Dobra cobertura de sinais (alguns dias só short ou só long, raros ambos).

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
# 2 targets binários — long_win e short_win
labeled = labeled.with_columns(
    (pl.col("label") == 1).cast(pl.Int8).alias("y_long"),
    (pl.col("label") == -1).cast(pl.Int8).alias("y_short"),
)
feature_cols = [c for c in labeled.columns if c not in feat.LAG_SAFE_EXCLUDE and c not in {"label","hit_bar","barrier_ret","upper_px","lower_px","y_long","y_short"}]
mat = labeled.select(["open_time","close","y_long","y_short","barrier_ret", *feature_cols]).drop_nulls(subset=feature_cols+["y_long","y_short"]).to_pandas()
mat["dt"] = mat["open_time"].apply(lambda ms: datetime.fromtimestamp(ms/1000, tz=timezone.utc))
mat["quarter"] = mat["dt"].dt.to_period("Q")

print(f"rows={len(mat)}  features={len(feature_cols)}")
print(f"base rate long_win  = {100*mat['y_long'].mean():.1f}%")
print(f"base rate short_win = {100*mat['y_short'].mean():.1f}%")

PARAMS = dict(objective="binary", metric="binary_logloss", verbose=-1, n_jobs=-1,
              learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, lambda_l2=0.5)
COST = 0.0008
HORIZON = 12
THR = 0.35

# %% [markdown]
# ## Walk-forward dual: LONG + SHORT independentes

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

    X_tr = mat.iloc[train_idx][feature_cols].values
    y_tr_long = mat.iloc[train_idx]["y_long"].values
    y_tr_short = mat.iloc[train_idx]["y_short"].values
    X_te = mat.iloc[test_use_idx][feature_cols].values
    ret_te = mat.iloc[test_use_idx]["barrier_ret"].values

    m_long = lgb.train(PARAMS, lgb.Dataset(X_tr, y_tr_long), num_boost_round=500)
    m_short = lgb.train(PARAMS, lgb.Dataset(X_tr, y_tr_short), num_boost_round=500)

    proba_long = m_long.predict(X_te)
    proba_short = m_short.predict(X_te)

    # LONG: ganha se barrier_ret > 0 (atingiu upper)
    # SHORT: ganha se barrier_ret < 0 (atingiu lower) — short captura -ret
    take_long = proba_long > THR
    take_short = proba_short > THR
    # Conflito: ambos sinalizam — anula (postura agnóstica)
    conflict = take_long & take_short
    take_long = take_long & ~conflict
    take_short = take_short & ~conflict

    # PnL long: barrier_ret - cost
    # PnL short: -barrier_ret - cost (lucra com queda)
    pnl_long = ret_te[take_long] - COST if take_long.any() else np.array([])
    pnl_short = (-ret_te[take_short]) - COST if take_short.any() else np.array([])

    def m(pnl, label):
        if len(pnl) < 1:
            return dict(label=label, n=0, total=0, win=0)
        return dict(label=label, n=len(pnl), total=pnl.sum(), win=(pnl>0).mean())

    ml = m(pnl_long, "long")
    ms = m(pnl_short, "short")

    # Combined strategy: long ou short consensual
    strat = np.zeros(len(ret_te))
    strat[take_long] = ret_te[take_long] - COST
    strat[take_short] = -ret_te[take_short] - COST
    n_combined = int(take_long.sum() + take_short.sum())
    total = strat.sum()
    sharpe = (strat.mean()/strat.std()) * np.sqrt(6*365) if strat.std() > 0 else 0

    print(
        f"  {str(q):>8s}  L={ml['n']:>4d}/PnL{100*ml['total']:+6.1f}%/win{100*ml['win']:.0f}%  "
        f"S={ms['n']:>4d}/PnL{100*ms['total']:+6.1f}%/win{100*ms['win']:.0f}%  "
        f"conflicts={int(conflict.sum()):>3d}  "
        f"combined={n_combined:>4d}/PnL{100*total:+6.1f}%/Sharpe{sharpe:+.2f}"
    )
    results.append({
        "quarter": str(q), "n_long": ml["n"], "n_short": ms["n"],
        "pnl_long": ml["total"], "pnl_short": ms["total"],
        "conflicts": int(conflict.sum()),
        "n_combined": n_combined, "pnl_combined": total, "sharpe": sharpe,
        "win_long": ml["win"], "win_short": ms["win"],
    })

# %% [markdown]
# ## Agregado

# %%
res = pd.DataFrame(results)
print("\n=== Agregado 14 trimestres ===")
print(f"{'lado':<12s}  {'sinais':>7s}  {'PnL total':>10s}  {'win%':>5s}")
print(f"  long-only    {res['n_long'].sum():>7d}  {100*res['pnl_long'].sum():>+9.1f}%  {100*res['win_long'].mean():>4.1f}%")
print(f"  short-only   {res['n_short'].sum():>7d}  {100*res['pnl_short'].sum():>+9.1f}%  {100*res['win_short'].mean():>4.1f}%")
print(f"  combined     {res['n_combined'].sum():>7d}  {100*res['pnl_combined'].sum():>+9.1f}%       —")
print(f"  conflicts:   {res['conflicts'].sum():>7d}  (filtrados)")
print(f"\nSharpe combinado (média por fold): {res['sharpe'].mean():+.2f}")
print(f"Folds combined positivos: {(res['pnl_combined']>0).sum()}/{len(res)}")

# %% [markdown]
# ## Mature folds (2025+)

# %%
mature = res.tail(6)
print(f"\n=== Folds maduros (últimos 6 trimestres) ===")
print(f"  long total:      {100*mature['pnl_long'].sum():+.1f}%  ({mature['n_long'].sum()} sinais)")
print(f"  short total:     {100*mature['pnl_short'].sum():+.1f}%  ({mature['n_short'].sum()} sinais)")
print(f"  combined total:  {100*mature['pnl_combined'].sum():+.1f}%")
print(f"  Sharpe (média):  {mature['sharpe'].mean():+.2f}")
