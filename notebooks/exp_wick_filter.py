# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # Experimento — Wick exhaustion filter (VSA)
#
# Hipótese (Velasques): vela com sombra superior longa = exaustão dos buyers = sinal fraco.
#
# Computa:
#   upper_wick = high - max(open, close)
#   body       = abs(open - close)
#   wick_ratio = upper_wick / body  (na vela t-1, ANTES do sinal em t)
#
# Filtra: se wick_ratio(t-1) > THR → REJEITA sinal em t.
#
# Testa thresholds: 0.3, 0.5, 0.7, 1.0
#
# Mantém: walk-forward expanding quarterly 2023Q1→2026Q2, threshold modelo=0.35, cost=0.0008.

# %%
from __future__ import annotations
from pathlib import Path
import sys, os, time
ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timezone
from pipeline import features as feat, labels as lab

TIMEFRAME = 240            # 4h
HORIZON_BARS = 12          # 48h
ATR_MULT = 3.0
COST = 0.0008
MODEL_THR = 0.35           # threshold de sinal (baseline v2)

# %% [markdown]
# ## 1. Build matrix v2 e adiciona wick_ratio(t-1)

# %%
df = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME, lag=1)
df = df.drop_nulls(subset=["atr_14"])

# OHLC RAW preservado por LAG_SAFE_EXCLUDE → wick computado na vela t.
# shift(1) para usar wick da vela ANTERIOR ao sinal.
df = df.with_columns(
    (pl.col("high") - pl.max_horizontal(pl.col("open"), pl.col("close"))).alias("_upper_wick"),
    (pl.col("close") - pl.col("open")).abs().alias("_body"),
).with_columns(
    pl.when(pl.col("_body") > 1e-9)
      .then(pl.col("_upper_wick") / pl.col("_body"))
      .otherwise(None)
      .shift(1)
      .alias("wick_ratio_prev"),
).drop(["_upper_wick", "_body"])

print(f"shape em 4h: {df.shape}")
print(f"range: {datetime.fromtimestamp(df['open_time'].min()/1000, tz=timezone.utc).date()} → "
      f"{datetime.fromtimestamp(df['open_time'].max()/1000, tz=timezone.utc).date()}")

# diag wick_ratio
wr = df["wick_ratio_prev"].drop_nulls().to_numpy()
print(f"\nwick_ratio_prev — n={len(wr)}, mean={wr.mean():.2f}, "
      f"p50={np.median(wr):.2f}, p75={np.percentile(wr,75):.2f}, p90={np.percentile(wr,90):.2f}")
for thr in [0.3, 0.5, 0.7, 1.0]:
    pct = (wr > thr).mean() * 100
    print(f"  % bars com wick_ratio > {thr}: {pct:.1f}%")

# %% [markdown]
# ## 2. Triple-barrier

# %%
labeled = lab.triple_barrier(df, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=HORIZON_BARS)
labeled = labeled.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y_bin"))

# %% [markdown]
# ## 3. Feature set (igual baseline — exclui wick_ratio_prev do modelo, só usa como filtro)

# %%
EXCLUDE_FROM_MODEL = {"wick_ratio_prev"}
feature_cols = [
    c for c in labeled.columns
    if c not in feat.LAG_SAFE_EXCLUDE
    and c not in {"label", "hit_bar", "barrier_ret", "upper_px", "lower_px", "y_bin"}
    and c not in EXCLUDE_FROM_MODEL
]
print(f"Features modelo: {len(feature_cols)}")

mat = labeled.select(
    ["open_time", "close", "y_bin", "barrier_ret", "wick_ratio_prev", *feature_cols]
).drop_nulls(subset=feature_cols + ["y_bin"]).to_pandas()
mat["dt"] = mat["open_time"].apply(lambda ms: datetime.fromtimestamp(ms/1000, tz=timezone.utc))
mat["quarter"] = mat["dt"].dt.to_period("Q")
print(f"Linhas usáveis: {len(mat)}")

# %% [markdown]
# ## 4. Walk-forward — gera probabilidades + retém wick_ratio_prev por amostra

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
    is_unbalance=False,
    verbose=-1,
    n_jobs=-1,
)
N_ROUNDS = 500

quarters = [q for q in sorted(mat["quarter"].unique()) if q.start_time.year >= 2023]
all_proba, all_y, all_ret, all_wick, all_dt = [], [], [], [], []

t_start = time.time()
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
    wick_te = mat.iloc[test_use_idx]["wick_ratio_prev"].values

    dtr = lgb.Dataset(X_tr, y_tr)
    model = lgb.train(PARAMS, dtr, num_boost_round=N_ROUNDS)
    proba = model.predict(X_te)

    all_proba.append(proba); all_y.append(y_te); all_ret.append(ret_te)
    all_wick.append(wick_te); all_dt.extend(mat.iloc[test_use_idx]["dt"].tolist())
    print(f"  {str(q):>8s}  tr={len(train_idx):>5d}  te={len(test_use_idx):>4d}")

print(f"\nWalk-forward total: {time.time()-t_start:.1f}s")

proba_all = np.concatenate(all_proba)
y_all = np.concatenate(all_y)
ret_all = np.concatenate(all_ret)
wick_all = np.concatenate(all_wick)
dt_all = pd.to_datetime(all_dt)

# %% [markdown]
# ## 5. Métricas — baseline (sem filtro) vs filtro wick em vários thresholds
#
# Sharpe anualizado: usa retornos por trade. ~6 bars/dia em 4h, anualização = sqrt(N_trades_per_year).
# Mas tradicionalmente em backtest de signal, usamos Sharpe(per trade) * sqrt(trades_per_year).
# Aqui aproximamos: sharpe = mean / std * sqrt(n_per_year), onde n_per_year = n_trades * 365 / dias_total.

# %%
def stats(pnl: np.ndarray, days: float) -> dict:
    n = len(pnl)
    if n < 2:
        return {"n": n, "sharpe": 0.0, "pf": 0.0, "win": 0.0, "avg": 0.0, "tot": 0.0}
    mu = pnl.mean()
    sd = pnl.std(ddof=1)
    trades_per_year = n * 365.0 / max(days, 1.0)
    sharpe = (mu / sd) * np.sqrt(trades_per_year) if sd > 0 else 0.0
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    pf = gains / losses if losses > 0 else np.inf
    win = (pnl > 0).mean()
    return {"n": n, "sharpe": sharpe, "pf": pf, "win": win, "avg": mu, "tot": pnl.sum()}

# Dias totais do pool
days_total = (dt_all.max() - dt_all.min()).days

# Baseline: take = proba > MODEL_THR
take_base = proba_all > MODEL_THR
pnl_base = ret_all[take_base] - COST
base_stats = stats(pnl_base, days_total)

print(f"\n=== Pool: {len(y_all)} amostras, base rate {100*y_all.mean():.1f}%, "
      f"dias={days_total} ===\n")
print(f"BASELINE (thr={MODEL_THR}, sem filtro wick):")
print(f"  n_signals={base_stats['n']}  Sharpe={base_stats['sharpe']:.2f}  "
      f"PF={base_stats['pf']:.2f}  win={100*base_stats['win']:.1f}%  "
      f"avg={100*base_stats['avg']:+.3f}%  tot={100*base_stats['tot']:+.2f}%\n")

# Sweep wick thresholds
rows = []
rows.append({
    "wick_thr": "none (baseline)", "n_signals": base_stats["n"], "pct_filtered": 0.0,
    "sharpe": base_stats["sharpe"], "pf": base_stats["pf"],
    "win": base_stats["win"], "avg": base_stats["avg"], "tot": base_stats["tot"],
    "d_sharpe": 0.0,
})

print(f"{'wick_thr':>12s}  {'n_sig':>6s}  {'%filt':>6s}  {'Sharpe':>7s}  "
      f"{'dSh':>6s}  {'PF':>5s}  {'win':>5s}  {'avg':>8s}  {'tot':>9s}")
print(f"  {'baseline':>10s}  {base_stats['n']:>5d}  {'  -  ':>5s}  "
      f"{base_stats['sharpe']:>6.2f}    -    {base_stats['pf']:>5.2f}  "
      f"{100*base_stats['win']:>4.1f}%  {100*base_stats['avg']:+7.3f}%  {100*base_stats['tot']:+8.2f}%")

for thr in [0.3, 0.5, 0.7, 1.0]:
    # Filtra: rejeita se wick_ratio_prev > thr. NaN considerado "não-exaustão" → mantém.
    wick_ok = ~(wick_all > thr)
    take = take_base & wick_ok
    n_base = take_base.sum()
    n_after = take.sum()
    n_filtered = n_base - n_after
    pct_filt = 100.0 * n_filtered / n_base if n_base > 0 else 0.0
    pnl = ret_all[take] - COST
    s = stats(pnl, days_total)
    d_sh = s["sharpe"] - base_stats["sharpe"]
    rows.append({
        "wick_thr": thr, "n_signals": s["n"], "pct_filtered": pct_filt,
        "sharpe": s["sharpe"], "pf": s["pf"], "win": s["win"],
        "avg": s["avg"], "tot": s["tot"], "d_sharpe": d_sh,
    })
    print(f"  > {thr:>5.2f}    {s['n']:>5d}  {pct_filt:>5.1f}%  "
          f"{s['sharpe']:>6.2f}  {d_sh:+5.2f}  {s['pf']:>5.2f}  "
          f"{100*s['win']:>4.1f}%  {100*s['avg']:+7.3f}%  {100*s['tot']:+8.2f}%")

# %% [markdown]
# ## 6. Recomendação

# %%
summary = pd.DataFrame(rows)
print("\n=== Sumário ===")
print(summary.to_string(index=False))

# Melhor threshold por Sharpe (excluindo baseline)
non_base = summary[summary["wick_thr"] != "none (baseline)"].copy()
best = non_base.loc[non_base["sharpe"].idxmax()]
print(f"\nMelhor wick_thr: {best['wick_thr']}  →  Sharpe={best['sharpe']:.2f}  "
      f"(Δ={best['d_sharpe']:+.2f} vs baseline {base_stats['sharpe']:.2f})")

if best["d_sharpe"] > 0.05 and best["n_signals"] >= 100:
    veredito = "INTEGRAR — ganho material no Sharpe, sample ainda razoável"
elif best["d_sharpe"] > 0:
    veredito = "MARGINAL — ganho pequeno, não compensa complexidade"
else:
    veredito = "DESCARTAR — filtro não melhora (ou piora) Sharpe"

print(f"\nVeredito: {veredito}")
