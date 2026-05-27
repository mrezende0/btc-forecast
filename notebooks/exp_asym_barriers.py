# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # EXP — Asymmetric Barriers Risk:Reward 1:2
#
# Hipótese: stop mais apertado (1.5×ATR) + target igual (3.0×ATR) melhora Sharpe
# mesmo com win rate menor, porque cada ganho vale 2× cada perda.
#
# Setup:
#   - upper_mult=3.0, lower_mult=1.5
#   - horizon=12 bars (48h)
#   - timeframe 4h
#   - walk-forward quarterly 2023Q1→2026Q2, purge+embargo=12 bars
#   - threshold sweep: 0.30, 0.35, 0.40, 0.45, 0.50
#   - custo round-trip 0.0008

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
from sklearn.metrics import precision_score
from datetime import datetime, timezone
from pipeline import features as feat, labels as lab

TIMEFRAME = 240  # 4h
HORIZON_BARS = 12  # 48h
UPPER_MULT = 3.0
LOWER_MULT = 1.5  # stop mais apertado → R:R 1:2
COST = 0.0015  # Binance taker 0.10% × 2 + slippage real

# %% [markdown]
# ## 1. Build matrix em 4h

# %%
df = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME, lag=1)
df = df.drop_nulls(subset=["atr_14"])
print(f"shape em 4h: {df.shape}")
print(
    f"range: {datetime.fromtimestamp(df['open_time'].min()/1000, tz=timezone.utc).date()} → "
    f"{datetime.fromtimestamp(df['open_time'].max()/1000, tz=timezone.utc).date()}"
)

# %% [markdown]
# ## 2. Triple-barrier ASIMÉTRICO (upper=3.0, lower=1.5)

# %%
labeled = lab.triple_barrier(
    df, upper_mult=UPPER_MULT, lower_mult=LOWER_MULT, horizon_bars=HORIZON_BARS
)
print(f"\nBarriers: upper=+{UPPER_MULT}×ATR | lower=-{LOWER_MULT}×ATR | R:R = 1:{UPPER_MULT/LOWER_MULT:.1f}")
print("\nDistribuição labels:")
total = labeled.height
for lbl, name in [(1, "LONG_WIN"), (0, "TIMEOUT"), (-1, "STOP")]:
    n = labeled.filter(pl.col("label") == lbl).height
    rt = labeled.filter(pl.col("label") == lbl)["barrier_ret"].mean()
    print(f"  {name:>8s}  {n:>5d}  ({100*n/total:>4.1f}%)  ret_µ={rt*100 if rt else 0:+.2f}%")

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

mat = labeled.select(["open_time", "close", "y_bin", "barrier_ret", *feature_cols]).drop_nulls(
    subset=feature_cols + ["y_bin"]
).to_pandas()
mat["dt"] = mat["open_time"].apply(lambda ms: datetime.fromtimestamp(ms/1000, tz=timezone.utc))
mat["quarter"] = mat["dt"].dt.to_period("Q")
print(f"Linhas usáveis: {len(mat)}")

# %% [markdown]
# ## 4. Walk-forward

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
REF_THRESHOLD = 0.35

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

    pred = (proba > REF_THRESHOLD).astype(int)
    take = pred == 1
    n_sig = take.sum()
    pnl_arr = ret_te[take] - COST if n_sig > 0 else np.array([])
    avg_pnl = pnl_arr.mean() if n_sig > 0 else 0
    tot_pnl = pnl_arr.sum() if n_sig > 0 else 0
    win_rate = (pnl_arr > 0).mean() if n_sig > 0 else 0
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

# %% [markdown]
# ## 5. Pool agregado + threshold sweep + Sharpe

# %%
proba_all = np.concatenate(all_proba)
y_all = np.concatenate(all_y)
ret_all = np.concatenate(all_ret)
dt_all = pd.Series(all_dt)

print(f"\n=== Pool agregado: {len(y_all)} amostras, base rate {100*y_all.mean():.1f}% ===\n")

# Sharpe annualization: assumir um sinal "ocupa" HORIZON_BARS de 4h = 48h.
# Trades/ano teórico se preencher 100% do tempo: 8760/48 = 182.5
# Sharpe annual = mean / std * sqrt(N_trades_per_year). Usaremos sqrt(252*6/HORIZON_BARS) ~ aproximação simples
# Padrão do projeto: Sharpe sobre PnL por sinal anualizado por sqrt(trades/ano), assumindo trades não overlapping.
# Para comparabilidade com baseline (Sharpe 0.88), uso: sharpe = mean(pnl)/std(pnl) * sqrt(N_signals_per_year)
# onde N_signals_per_year é o número médio de sinais por ano observado.

span_years = (dt_all.max() - dt_all.min()).days / 365.25
print(f"Span coberto: {span_years:.2f} anos\n")

print(f"{'thr':>5s}  {'n_sig':>6s}  {'rate':>5s}  {'prec':>5s}  {'win':>5s}  {'avg':>8s}  {'tot':>9s}  {'PF':>5s}  {'Sharpe':>7s}")
sweep_rows = []
for thr in [0.30, 0.35, 0.40, 0.45, 0.50]:
    take = proba_all > thr
    n = int(take.sum())
    if n < 10:
        print(f"  >{thr:.2f}      {n:>5d}   (insufficient)")
        continue
    pnl = ret_all[take] - COST
    win = (pnl > 0).mean()
    avg = pnl.mean()
    tot = pnl.sum()
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    pf = gains / losses if losses > 0 else np.inf
    sig_per_year = n / span_years if span_years > 0 else 0
    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(sig_per_year) if pnl.std() > 0 else 0
    pred = take.astype(int)
    prec = precision_score(y_all, pred, zero_division=0)
    sweep_rows.append({
        "threshold": thr, "n_signals": n, "rate": n/len(y_all),
        "precision": prec, "win_rate": win, "avg_pnl": avg,
        "tot_pnl": tot, "pf": pf, "sharpe": sharpe, "sig_per_year": sig_per_year,
    })
    print(
        f"  >{thr:.2f}    {n:>5d}  {100*n/len(y_all):>4.1f}%  {100*prec:>4.1f}%  "
        f"{100*win:>4.1f}%  {100*avg:+.3f}%  {100*tot:+7.2f}%  {pf:>5.2f}  {sharpe:>7.3f}"
    )

sweep = pd.DataFrame(sweep_rows)

# %% [markdown]
# ## 6. Comparação vs baseline v2

# %%
BASELINE = {"sharpe": 0.88, "pf": 1.14, "win": 0.53, "n_signals": 787, "threshold": 0.35}
print(f"\n=== Baseline v2 (symmetric 3.0/3.0): Sharpe={BASELINE['sharpe']}, PF={BASELINE['pf']}, "
      f"win={100*BASELINE['win']:.0f}%, n_sig={BASELINE['n_signals']}, thr={BASELINE['threshold']} ===")

ref = sweep[sweep["threshold"] == REF_THRESHOLD].iloc[0] if (sweep["threshold"] == REF_THRESHOLD).any() else None
if ref is not None:
    print(f"\n=== EXP asymmetric (3.0/1.5) @ thr={REF_THRESHOLD}: "
          f"Sharpe={ref['sharpe']:.3f}, PF={ref['pf']:.3f}, "
          f"win={100*ref['win_rate']:.1f}%, n_sig={int(ref['n_signals'])}, "
          f"totPnL={100*ref['tot_pnl']:+.2f}% ===")
    print(f"\nDelta Sharpe: {ref['sharpe'] - BASELINE['sharpe']:+.3f}")
    print(f"Delta PF:     {ref['pf'] - BASELINE['pf']:+.3f}")
    print(f"Delta win:    {100*(ref['win_rate'] - BASELINE['win']):+.1f}pp")

# best threshold by Sharpe
if len(sweep) > 0:
    best = sweep.loc[sweep["sharpe"].idxmax()]
    print(f"\n=== Best threshold by Sharpe: thr={best['threshold']:.2f} → "
          f"Sharpe={best['sharpe']:.3f}, PF={best['pf']:.3f}, win={100*best['win_rate']:.1f}%, "
          f"n_sig={int(best['n_signals'])}, totPnL={100*best['tot_pnl']:+.2f}% ===")

# %% [markdown]
# ## 7. Sumário por trimestre

# %%
res = pd.DataFrame(results)
print(f"\nFolds: {len(res)}  |  positivos: {(res['tot_pnl']>0).sum()}/{len(res)} ({100*(res['tot_pnl']>0).mean():.0f}%)")
print(f"PnL acumulado total @ thr={REF_THRESHOLD}: {100*res['tot_pnl'].sum():+.2f}%")
print(f"Win rate médio:                            {100*res['win_rate'].mean():.1f}%")
print(f"Precision média:                           {100*res['precision'].mean():.1f}%")
print(f"Avg PnL/sinal médio:                       {100*res['avg_pnl'].mean():+.3f}%")
print(f"Sinais/trim médio:                         {res['n_signals'].mean():.0f}")
