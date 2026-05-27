# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # 08 — Model v3: v2 + sentiment features
#
# Repete walk-forward de v2 (4h, binary, interactions, regime) com sentiment news
# adicionado. Compara delta vs v2.
#
# Pré-requisitos:
#   1. Chunks GDELT mergeados: python -m pipeline.news_merge
#   2. FinBERT scoring rodado: python -m pipeline.sentiment_agg --recompute-all
#   3. data/sentiment_daily.parquet existe e cobre 2021+

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

# Verifica pré-requisitos
sent_path = Path("data/sentiment_daily.parquet")
if not sent_path.exists():
    raise FileNotFoundError(
        f"{sent_path} não existe. Rode primeiro:\n"
        "  python -m pipeline.news_merge\n"
        "  python -m pipeline.sentiment_agg --recompute-all"
    )

sd = pl.read_parquet(sent_path)
print(f"Sentiment cobertura: {sd['date'].min()} → {sd['date'].max()}  ({sd.height} dias)")
print(sd.head(3))

# %% [markdown]
# ## 1. Build matrix com sentiment

# %%
df = feat.build_v2_from_parquets(timeframe_min=240, lag=1).drop_nulls(subset=["atr_14"])
print(f"shape: {df.shape}")
sentiment_cols = [c for c in df.columns if c in {"news_count", "net_sentiment", "news_count_z30", "net_sentiment_z30"}]
print(f"Sentiment features presentes: {sentiment_cols}")
if not sentiment_cols:
    raise RuntimeError("Sentiment não entrou nas features. Verificar add_sentiment_news em features.py")

# %% [markdown]
# ## 2. Labels + train matrix

# %%
labeled = lab.triple_barrier(df, upper_mult=3.0, lower_mult=3.0, horizon_bars=12)
labeled = labeled.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y"))
feature_cols = [c for c in labeled.columns if c not in feat.LAG_SAFE_EXCLUDE and c not in {"label","hit_bar","barrier_ret","upper_px","lower_px","y"}]
mat = labeled.select(["open_time","close","y","barrier_ret", *feature_cols]).drop_nulls(subset=feature_cols+["y"]).to_pandas()
mat["dt"] = mat["open_time"].apply(lambda ms: datetime.fromtimestamp(ms/1000, tz=timezone.utc))
mat["quarter"] = mat["dt"].dt.to_period("Q")
print(f"Linhas usáveis: {len(mat)}  features: {len(feature_cols)}")

# %% [markdown]
# ## 3. Walk-forward v3

# %%
PARAMS = dict(
    objective="binary", metric="binary_logloss", verbose=-1, n_jobs=-1,
    learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, lambda_l2=0.5,
)
COST = 0.0015  # Binance taker 0.10% × 2 + slippage real
HORIZON = 12
THR = 0.35

quarters = [q for q in sorted(mat["quarter"].unique()) if q.start_time.year >= 2023]
all_proba, all_y, all_ret = [], [], []

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
    y_te = mat.iloc[test_use_idx]["y"].values
    ret_te = mat.iloc[test_use_idx]["barrier_ret"].values
    model = lgb.train(PARAMS, lgb.Dataset(X_tr, y_tr), num_boost_round=500)
    proba = model.predict(X_te)
    all_proba.append(proba)
    all_y.append(y_te)
    all_ret.append(ret_te)
    take = proba > THR
    pnl = (ret_te[take] - COST).sum() if take.any() else 0
    print(f"  {str(q):>8s}  n_sig={int(take.sum()):>4d}  totPnL={100*pnl:+.2f}%")

last_model = model

# %% [markdown]
# ## 4. Agregado + comparação v2 vs v3

# %%
proba = np.concatenate(all_proba)
y = np.concatenate(all_y)
ret = np.concatenate(all_ret)

print("\n=== v3 (com sentiment) ===")
print(f"{'thr':>5s}  {'n':>5s}  {'win%':>5s}  {'total':>7s}  {'Sharpe':>7s}  {'PF':>5s}")
for thr in [0.30, 0.35, 0.40, 0.45]:
    take = proba > thr
    n = take.sum()
    if n < 10: continue
    strat = np.where(take, ret - COST, 0.0)
    pnl_nz = strat[take]
    win = (pnl_nz > 0).mean()
    total = np.cumprod(1 + strat)[-1] - 1
    sharpe = (strat.mean()/strat.std()) * np.sqrt(6*365) if strat.std()>0 else 0
    pf = pnl_nz[pnl_nz>0].sum() / max(1e-9, -pnl_nz[pnl_nz<0].sum())
    print(f"  {thr:.2f}  {n:>5d}  {100*win:>4.1f}%  {100*total:+6.1f}%  {sharpe:+6.2f}  {pf:>5.2f}")

# %% [markdown]
# ## 5. Top features (último fold) — sentiment entrou no top?

# %%
imp = pd.DataFrame({
    "feature": feature_cols,
    "gain": last_model.feature_importance(importance_type="gain"),
}).sort_values("gain", ascending=False)
print("\nTop 25 features por GAIN:")
print(imp.head(25).to_string(index=False))

print("\nRanking das features de sentiment:")
sent_features = ["news_count", "net_sentiment", "news_count_z30", "net_sentiment_z30"]
for sf in sent_features:
    rank_row = imp[imp["feature"] == sf]
    if not rank_row.empty:
        rank = (imp["feature"].tolist().index(sf) + 1)
        gain = rank_row["gain"].iloc[0]
        print(f"  {sf:25s}  rank #{rank:>2d}  gain={gain:>10.0f}")
