# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # exp_ensemble — LightGBM + XGBoost ensemble
#
# Compara baseline LGB (Sharpe 0.88 @ thr 0.35) com:
#   - XGB sozinho
#   - Ensemble média simples (LGB + XGB) / 2
#   - Ensemble ponderado 0.6*LGB + 0.4*XGB
#   - Ensemble MAX(LGB, XGB)
#
# Mesma feature matrix v2, walk-forward expanding quarterly, purge=12.

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
import xgboost as xgb
from sklearn.metrics import precision_score
from datetime import datetime, timezone
from pipeline import features as feat, labels as lab
import time

TIMEFRAME = 240   # 4h
HORIZON_BARS = 12 # 48h purge
ATR_MULT = 3.0
COST = 0.0015  # Binance taker 0.10% × 2 + slippage real
THRESHOLD = 0.35
BARS_PER_YEAR = 6 * 365  # 4h bars

# %% [markdown]
# ## 1. Build matrix

# %%
df = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME, lag=1)
df = df.drop_nulls(subset=["atr_14"])
print(f"shape 4h: {df.shape}")

labeled = lab.triple_barrier(df, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=HORIZON_BARS)
labeled = labeled.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y_bin"))

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
print(f"Base rate LONG_WIN: {100*mat['y_bin'].mean():.1f}%")

# %% [markdown]
# ## 2. Params

# %%
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
    is_unbalance=False,
    verbose=-1,
    n_jobs=-1,
)
LGB_ROUNDS = 500

XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    max_depth=6,
    learning_rate=0.05,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=0.5,
    tree_method="hist",
    n_jobs=-1,
    verbosity=0,
)

# %% [markdown]
# ## 3. Walk-forward

# %%
quarters = [q for q in sorted(mat["quarter"].unique()) if q.start_time.year >= 2023]

all_proba_lgb = []
all_proba_xgb = []
all_y = []
all_ret = []
all_dt = []

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
    # LGB
    dtr = lgb.Dataset(X_tr, y_tr)
    lgb_model = lgb.train(LGB_PARAMS, dtr, num_boost_round=LGB_ROUNDS)
    p_lgb = lgb_model.predict(X_te)
    t_lgb = time.time() - t0

    # XGB
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(X_tr, y_tr)
    p_xgb = xgb_model.predict_proba(X_te)[:, 1]
    t_xgb = time.time() - t0

    all_proba_lgb.append(p_lgb)
    all_proba_xgb.append(p_xgb)
    all_y.append(y_te)
    all_ret.append(ret_te)
    all_dt.extend(mat.iloc[test_use_idx]["dt"].tolist())

    print(
        f"{str(q):>8s}  tr={len(train_idx):>5d}  te={len(test_use_idx):>4d}  "
        f"LGB={t_lgb:.1f}s  XGB={t_xgb:.1f}s  "
        f"corr(lgb,xgb)={np.corrcoef(p_lgb, p_xgb)[0,1]:.3f}"
    )

# %% [markdown]
# ## 4. Avaliação por combinação

# %%
proba_lgb = np.concatenate(all_proba_lgb)
proba_xgb = np.concatenate(all_proba_xgb)
y_all = np.concatenate(all_y)
ret_all = np.concatenate(all_ret)
dt_all = pd.to_datetime(all_dt, utc=True)

combos = {
    "LGB only":      proba_lgb,
    "XGB only":      proba_xgb,
    "Mean (50/50)":  (proba_lgb + proba_xgb) / 2,
    "Weighted .6/.4":(0.6 * proba_lgb + 0.4 * proba_xgb),
    "Max":           np.maximum(proba_lgb, proba_xgb),
}

def evaluate(proba: np.ndarray, name: str) -> dict:
    take = proba > THRESHOLD
    n_sig = int(take.sum())
    if n_sig < 5:
        return {"name": name, "n_sig": n_sig, "sharpe": np.nan, "pnl_tot": 0,
                "win_rate": 0, "maxdd": 0, "prec": 0, "avg_pnl": 0}
    pnl = ret_all[take] - COST
    win = (pnl > 0).mean()
    avg = pnl.mean()
    tot = pnl.sum()
    prec = precision_score(y_all, take.astype(int), zero_division=0)

    # Sharpe: build equity curve quarterly-aligned
    # série temporal: ordena por dt_all -> pega só sinais -> retorno por trade
    sig_dt = dt_all[take]
    sig_ret = pnl
    order = np.argsort(sig_dt.values)
    sig_dt = sig_dt[order]
    sig_ret = sig_ret[order]
    # Annualized sharpe per trade: usa média/std do retorno por trade
    # mas escalamos por nº de trades/ano
    if len(sig_ret) > 1 and sig_ret.std() > 0:
        years = (sig_dt.max() - sig_dt.min()).days / 365.25
        trades_per_year = len(sig_ret) / max(years, 1e-6)
        sharpe = (sig_ret.mean() / sig_ret.std()) * np.sqrt(trades_per_year)
    else:
        sharpe = np.nan

    # MaxDD da equity curve
    eq = np.cumsum(sig_ret)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak)
    maxdd = dd.min() if len(dd) > 0 else 0

    return {
        "name": name, "n_sig": n_sig, "sharpe": sharpe, "pnl_tot": tot,
        "win_rate": win, "maxdd": maxdd, "prec": prec, "avg_pnl": avg,
    }

results = [evaluate(p, n) for n, p in combos.items()]
res = pd.DataFrame(results)

print(f"\n=== Pool: {len(y_all)} amostras, base rate {100*y_all.mean():.1f}%, threshold {THRESHOLD} ===\n")
print(f"{'combo':<18s}  {'n_sig':>6s}  {'sharpe':>7s}  {'pnl_tot':>9s}  {'win':>6s}  {'maxdd':>8s}  {'prec':>6s}  {'avg':>8s}")
for r in results:
    print(
        f"{r['name']:<18s}  {r['n_sig']:>6d}  {r['sharpe']:>7.3f}  "
        f"{100*r['pnl_tot']:>+8.2f}%  {100*r['win_rate']:>5.1f}%  "
        f"{100*r['maxdd']:>+7.2f}%  {100*r['prec']:>5.1f}%  {100*r['avg_pnl']:>+7.3f}%"
    )

# %% [markdown]
# ## 5. Threshold sweep do melhor ensemble

# %%
best_name = max(results, key=lambda r: r['sharpe'] if not np.isnan(r['sharpe']) else -np.inf)['name']
best_proba = combos[best_name]
print(f"\n=== Threshold sweep — {best_name} ===")
print(f"{'thr':>6s}  {'n':>5s}  {'sharpe':>7s}  {'pnl':>9s}  {'win':>6s}")
for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    take = best_proba > thr
    n = int(take.sum())
    if n < 5:
        continue
    pnl = ret_all[take] - COST
    if pnl.std() > 0:
        sig_dt = dt_all[take]
        order = np.argsort(sig_dt.values)
        sd = sig_dt[order]
        years = (sd.max() - sd.min()).days / 365.25
        tpy = n / max(years, 1e-6)
        sh = (pnl.mean() / pnl.std()) * np.sqrt(tpy)
    else:
        sh = np.nan
    print(f"  {thr:.2f}  {n:>5d}  {sh:>7.3f}  {100*pnl.sum():>+7.2f}%  {100*(pnl>0).mean():>5.1f}%")

# %% [markdown]
# ## 6. Sumário final

# %%
lgb_sharpe = next(r['sharpe'] for r in results if r['name'] == 'LGB only')
print(f"\n=== Delta Sharpe vs LGB baseline ({lgb_sharpe:.3f}) ===")
for r in results:
    if r['name'] == 'LGB only':
        continue
    delta = r['sharpe'] - lgb_sharpe
    print(f"  {r['name']:<18s}  sharpe={r['sharpe']:.3f}  Δ={delta:+.3f}")
