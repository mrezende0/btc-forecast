# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # 05 — Modelo + Walk-forward honesto
#
# LightGBM 3-class (LONG_WIN / TIMEOUT / STOP) com:
#   - Walk-forward expanding (treina passado, prevê futuro)
#   - Purge: descarta últimas 32 velas do treino (horizonte do label)
#   - Embargo: descarta primeiras 32 velas do teste (autocorrelação)
#   - Folds trimestrais cobrindo 2023-Q1 → presente
#   - Hyperparams fixos (sem hyperopt nesta versão — vem depois)
#
# Métricas:
#   - Accuracy por classe
#   - Precision em LONG_WIN (o que importa pra entrar comprado)
#   - Log loss
#   - Retorno esperado por sinal (cumulative se modelo virar estratégia)

# %%
from __future__ import annotations
from pathlib import Path
import sys, os
ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, confusion_matrix
from datetime import datetime, timezone

HORIZON_BARS = 32  # mesmo do labeling — pra purge
EMBARGO_BARS = 32  # mesmo — pra embargo

# %% [markdown]
# ## 1. Carrega matriz

# %%
mat = pl.read_parquet("data/training_matrix.parquet").sort("open_time")
print(f"shape: {mat.shape}")
print(f"range: {datetime.fromtimestamp(mat['open_time'].min()/1000, tz=timezone.utc).date()} → {datetime.fromtimestamp(mat['open_time'].max()/1000, tz=timezone.utc).date()}")

feature_cols = [c for c in mat.columns if c not in {"open_time", "close", "label", "hit_bar", "barrier_ret"}]
print(f"features: {len(feature_cols)}")

# %% [markdown]
# ## 2. Define folds trimestrais

# %%
mat_pd = mat.to_pandas()
mat_pd["dt"] = mat_pd["open_time"].apply(lambda ms: datetime.fromtimestamp(ms/1000, tz=timezone.utc))
mat_pd["quarter"] = mat_pd["dt"].dt.to_period("Q")

# Folds: testar do primeiro trimestre que tem >= 2 anos de histórico antes
all_quarters = sorted(mat_pd["quarter"].unique())
test_quarters = [q for q in all_quarters if q.start_time.year >= 2023]
print(f"Trimestres de teste: {len(test_quarters)}  ({test_quarters[0]} → {test_quarters[-1]})")

# %% [markdown]
# ## 3. Walk-forward

# %%
LGB_PARAMS = dict(
    objective="multiclass",
    num_class=3,
    metric="multi_logloss",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l2=0.1,
    verbose=-1,
    n_jobs=-1,
)
N_ROUNDS = 400

# Mapeia labels {-1, 0, 1} → {0, 1, 2} para LGB multiclass
LBL_MAP = {-1: 0, 0: 1, 1: 2}
INV_MAP = {0: -1, 1: 0, 2: 1}
mat_pd["y"] = mat_pd["label"].map(LBL_MAP)

results = []
shap_top = None
import time

for q in test_quarters:
    test_mask = mat_pd["quarter"] == q
    test_idx = mat_pd.index[test_mask].tolist()
    if not test_idx:
        continue

    test_start = test_idx[0]
    train_end = test_start - HORIZON_BARS  # purge
    test_use_start = test_start + EMBARGO_BARS  # embargo
    if train_end < 1000:
        continue  # warm-up insuficiente
    if test_use_start >= test_idx[-1]:
        continue

    train_idx = list(range(0, train_end))
    test_use_idx = [i for i in test_idx if i >= test_use_start]

    X_tr = mat_pd.iloc[train_idx][feature_cols].values
    y_tr = mat_pd.iloc[train_idx]["y"].values
    X_te = mat_pd.iloc[test_use_idx][feature_cols].values
    y_te = mat_pd.iloc[test_use_idx]["y"].values
    ret_te = mat_pd.iloc[test_use_idx]["barrier_ret"].values

    t0 = time.time()
    dtr = lgb.Dataset(X_tr, y_tr)
    model = lgb.train(LGB_PARAMS, dtr, num_boost_round=N_ROUNDS)
    dt = time.time() - t0

    proba = model.predict(X_te)
    pred = np.argmax(proba, axis=1)

    acc = accuracy_score(y_te, pred)
    ll = log_loss(y_te, proba, labels=[0,1,2])
    prec_long = precision_score(y_te, pred, labels=[2], average="macro", zero_division=0)
    rec_long = recall_score(y_te, pred, labels=[2], average="macro", zero_division=0)

    # Métrica de PnL: entra long quando pred==LONG_WIN (classe 2)
    take_long = pred == 2
    # Retorno esperado: barrier_ret quando label era +1 ou -1, ou aproximação no timeout
    n_signals = take_long.sum()
    # Custo round-trip 0.08% por trade
    cost = 0.0008
    pnl_long = ret_te[take_long] - cost if n_signals > 0 else np.array([])
    avg_pnl = pnl_long.mean() if n_signals > 0 else 0
    total_pnl = pnl_long.sum() if n_signals > 0 else 0
    win_rate_signal = (pnl_long > 0).mean() if n_signals > 0 else 0

    results.append({
        "quarter": str(q),
        "train_size": len(train_idx),
        "test_size": len(test_use_idx),
        "acc": acc,
        "log_loss": ll,
        "prec_long": prec_long,
        "rec_long": rec_long,
        "signals_long": int(n_signals),
        "signal_rate": n_signals / len(test_use_idx),
        "avg_pnl": avg_pnl,
        "total_pnl": total_pnl,
        "win_rate_signal": win_rate_signal,
        "train_secs": dt,
    })

    print(
        f"{str(q):>8s}  tr={len(train_idx):>6d}  te={len(test_use_idx):>5d}  "
        f"acc={acc:.3f}  ll={ll:.3f}  prec_L={prec_long:.3f}  "
        f"sinais={int(n_signals):>4d}({100*n_signals/len(test_use_idx):>4.1f}%)  "
        f"avgPnL={avg_pnl*100:+.3f}%  totPnL={total_pnl*100:+.2f}%  win%={win_rate_signal*100:.1f}  ({dt:.1f}s)"
    )

    # Salva model do último fold pra SHAP
    last_model = model
    last_X_te = X_te
    last_y_te = y_te
    last_pred = pred

# %% [markdown]
# ## 4. Resumo agregado

# %%
import pandas as pd
res = pd.DataFrame(results)
print("\n=== Sumário walk-forward ===")
print(f"Folds: {len(res)}")
print(f"Acc média:        {res['acc'].mean():.3f}  (±{res['acc'].std():.3f})")
print(f"Log loss média:   {res['log_loss'].mean():.3f}")
print(f"Prec LONG média:  {res['prec_long'].mean():.3f}")
print(f"Sinais/trim:      {res['signals_long'].mean():.0f}")
print(f"Taxa sinal:       {100*res['signal_rate'].mean():.1f}%")
print(f"PnL médio/sinal:  {100*res['avg_pnl'].mean():+.3f}%  (custo já descontado)")
print(f"PnL total acumulado: {100*res['total_pnl'].sum():+.2f}%")
print(f"% folds com PnL+: {100*(res['total_pnl']>0).mean():.0f}%")

# %% [markdown]
# ## 5. Feature importance (último fold)

# %%
imp = pd.DataFrame({
    "feature": feature_cols,
    "gain": last_model.feature_importance(importance_type="gain"),
    "split": last_model.feature_importance(importance_type="split"),
}).sort_values("gain", ascending=False)
print("\nTop 15 features por GAIN (último fold):")
print(imp.head(15).to_string(index=False))

# %% [markdown]
# ## 6. Matriz de confusão último fold

# %%
print("\nConfusion matrix (último fold):")
print("                pred_STOP  pred_TIMEOUT  pred_LONG")
cm = confusion_matrix(last_y_te, last_pred, labels=[0, 1, 2])
for i, name in enumerate(["true_STOP", "true_TIMEOUT", "true_LONG"]):
    print(f"  {name:>14s}  {cm[i,0]:>8d}  {cm[i,1]:>11d}  {cm[i,2]:>10d}")
