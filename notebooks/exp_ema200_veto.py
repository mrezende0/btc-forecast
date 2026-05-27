# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # Experimento — EMA200 daily como veto hard (hipótese Velasques)
#
# Treina v2 normal, mas pós-filtra sinais: só executa long se close_4h > EMA200_daily(prev_day).
# Compara baseline (sem filtro) vs com filtro.

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
from sklearn.metrics import precision_score
from datetime import datetime, timezone
from pipeline import features as feat, labels as lab

TIMEFRAME = 240  # 4h
HORIZON_BARS = 12  # 48h
ATR_MULT = 3.0
COST = 0.0008
THRESHOLD = 0.35

# %% [markdown]
# ## 1. Build matrix v2 4h (mesmo do baseline)

# %%
df = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME, lag=1)
df = df.drop_nulls(subset=["atr_14"])
print(f"shape 4h: {df.shape}")

# %% [markdown]
# ## 2. Calcula EMA200 daily a partir do ohlcv_15m bruto
#  - Agrega close do último bar de cada dia UTC
#  - ema200 = ewm(span=200, adjust=False)
#  - Aplica lag de 1 dia (usa EMA200 do dia ANTERIOR no bar de 4h)

# %%
ohlcv_15m = pl.read_parquet("data/ohlcv_15m.parquet")
o15 = ohlcv_15m.to_pandas().sort_values("open_time").reset_index(drop=True)
o15["dt_utc"] = pd.to_datetime(o15["open_time"], unit="ms", utc=True)
o15["date_utc"] = o15["dt_utc"].dt.floor("D")

# close diario = ultimo bar do dia
daily = o15.groupby("date_utc", as_index=False).agg(close=("close", "last"))
daily = daily.sort_values("date_utc").reset_index(drop=True)
daily["ema200"] = daily["close"].ewm(span=200, adjust=False).mean()
# lag de 1 dia: ema200 efetiva no dia D = ema200 calculada com close do dia D-1
daily["ema200_lag1"] = daily["ema200"].shift(1)
print(f"daily bars: {len(daily)}  range {daily['date_utc'].min().date()} -> {daily['date_utc'].max().date()}")
print(daily.tail(5).to_string(index=False))

# %% [markdown]
# ## 3. As-of join EMA200 no DF de 4h
#  - Para cada bar de 4h em data D, usa ema200_lag1 do dia D (que ja eh ema do dia D-1)

# %%
df_pd = df.to_pandas().sort_values("open_time").reset_index(drop=True)
df_pd["dt_utc"] = pd.to_datetime(df_pd["open_time"], unit="ms", utc=True)
df_pd["date_utc"] = df_pd["dt_utc"].dt.floor("D")
df_pd = df_pd.merge(daily[["date_utc", "ema200_lag1"]], on="date_utc", how="left")
print(f"merged: {len(df_pd)}  null ema200_lag1: {df_pd['ema200_lag1'].isna().sum()}")

# converte de volta para polars com a nova coluna
df = pl.from_pandas(df_pd.drop(columns=["dt_utc", "date_utc"]))

# %% [markdown]
# ## 4. Triple barrier + binary

# %%
labeled = lab.triple_barrier(df, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=HORIZON_BARS)
labeled = labeled.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y_bin"))

# Features: mesmo set v2, NAO inclui ema200_lag1 nas features (so usa como post-filter)
EXTRA_EXCLUDE = {"label", "hit_bar", "barrier_ret", "upper_px", "lower_px", "y_bin", "ema200_lag1"}
feature_cols = [
    c for c in labeled.columns
    if c not in feat.LAG_SAFE_EXCLUDE and c not in EXTRA_EXCLUDE
]
print(f"Features: {len(feature_cols)}")

mat = labeled.select(["open_time", "close", "y_bin", "barrier_ret", "ema200_lag1", *feature_cols]).drop_nulls(
    subset=feature_cols + ["y_bin"]
).to_pandas()
mat["dt"] = mat["open_time"].apply(lambda ms: datetime.fromtimestamp(ms / 1000, tz=timezone.utc))
mat["quarter"] = mat["dt"].dt.to_period("Q")
mat = mat.reset_index(drop=True)
print(f"Linhas usaveis: {len(mat)}")
print(f"null ema200_lag1 apos drop features: {mat['ema200_lag1'].isna().sum()}")

# %% [markdown]
# ## 5. Walk-forward expanding quarterly 2023Q1 -> 2026Q2

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
all_proba = []
all_y = []
all_ret = []
all_dt = []
all_close = []
all_ema = []

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
    close_te = mat.iloc[test_use_idx]["close"].values
    ema_te = mat.iloc[test_use_idx]["ema200_lag1"].values

    t0 = time.time()
    dtr = lgb.Dataset(X_tr, y_tr)
    model = lgb.train(PARAMS, dtr, num_boost_round=N_ROUNDS)
    dt = time.time() - t0

    proba = model.predict(X_te)
    all_proba.append(proba)
    all_y.append(y_te)
    all_ret.append(ret_te)
    all_dt.extend(mat.iloc[test_use_idx]["dt"].tolist())
    all_close.append(close_te)
    all_ema.append(ema_te)

    print(f"{str(q):>8s}  tr={len(train_idx):>5d}  te={len(test_use_idx):>4d}  ({dt:.1f}s)")

# %% [markdown]
# ## 6. Comparacao baseline vs filtro EMA200

# %%
proba_all = np.concatenate(all_proba)
y_all = np.concatenate(all_y)
ret_all = np.concatenate(all_ret)
close_all = np.concatenate(all_close)
ema_all = np.concatenate(all_ema)
dt_all = pd.to_datetime(all_dt)

# horizon em dias (HORIZON_BARS * 4h = 48h = 2 dias)
horizon_days = HORIZON_BARS * (TIMEFRAME / 60) / 24

def metrics_for_mask(mask: np.ndarray, label: str):
    n = int(mask.sum())
    if n == 0:
        print(f"{label}: 0 sinais")
        return None
    pnl = ret_all[mask] - COST
    win = (pnl > 0).mean()
    avg = pnl.mean()
    tot = pnl.sum()
    # equity curve por ordem temporal dos sinais
    order = np.argsort(dt_all[mask].values)
    pnl_ord = pnl[order]
    equity = np.cumsum(pnl_ord)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = dd.min()
    # Sharpe anualizado: trades sao discretos, escala por sqrt(N_per_year)
    # span temporal real coberto
    span_days = (dt_all[mask].max() - dt_all[mask].min()).total_seconds() / 86400
    trades_per_year = n / max(span_days / 365.25, 1e-9)
    if pnl.std(ddof=1) > 0:
        sharpe = (pnl.mean() / pnl.std(ddof=1)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    pf = gains / losses if losses > 0 else np.inf
    print(f"{label}")
    print(f"  sinais        : {n}")
    print(f"  win rate      : {100*win:.1f}%")
    print(f"  avg pnl/sinal : {100*avg:+.3f}%")
    print(f"  tot pnl       : {100*tot:+.2f}%")
    print(f"  profit factor : {pf:.2f}")
    print(f"  max drawdown  : {100*max_dd:.2f}%")
    print(f"  sharpe (anu)  : {sharpe:.2f}")
    print(f"  span dias     : {span_days:.0f}  ({trades_per_year:.0f} sinais/ano)")
    return dict(n=n, win=win, avg=avg, tot=tot, pf=pf, max_dd=max_dd, sharpe=sharpe)

print(f"\nPool agregado: {len(y_all)} amostras, base rate {100*y_all.mean():.1f}%")
print(f"NaN em ema200: {np.isnan(ema_all).sum()}")

print("\n" + "=" * 60)
print("BASELINE (sem filtro EMA200) — threshold > 0.35")
print("=" * 60)
mask_base = proba_all > THRESHOLD
base = metrics_for_mask(mask_base, "baseline")

print("\n" + "=" * 60)
print("COM FILTRO EMA200 (close > ema200_lag1)")
print("=" * 60)
mask_uptrend = ~np.isnan(ema_all) & (close_all > ema_all)
mask_filt = mask_base & mask_uptrend
n_blocked = int(mask_base.sum() - mask_filt.sum())
print(f"sinais bloqueados pelo filtro: {n_blocked} ({100*n_blocked/max(mask_base.sum(),1):.1f}%)")
filt = metrics_for_mask(mask_filt, "ema200_veto")

# %% [markdown]
# ## 7. Resumo final

# %%
print("\n" + "=" * 60)
print("DELTA")
print("=" * 60)
if base and filt:
    print(f"Sharpe   : {base['sharpe']:.2f} -> {filt['sharpe']:.2f}  (delta {filt['sharpe']-base['sharpe']:+.2f})")
    print(f"PF       : {base['pf']:.2f} -> {filt['pf']:.2f}")
    print(f"Win      : {100*base['win']:.1f}% -> {100*filt['win']:.1f}%")
    print(f"MaxDD    : {100*base['max_dd']:.2f}% -> {100*filt['max_dd']:.2f}%")
    print(f"Sinais   : {base['n']} -> {filt['n']} (-{base['n']-filt['n']})")
    print(f"Tot PnL  : {100*base['tot']:+.2f}% -> {100*filt['tot']:+.2f}%")
