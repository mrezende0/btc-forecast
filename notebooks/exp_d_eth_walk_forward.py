"""exp_d_eth_walk_forward — Multi-asset Fase 1.

Roda walk-forward MID-only no ETH com mesma config validada do BTC:
  thr=0.35, no_bear=-0.05, COST=0.0015, FULL sizing, uniqueness weights.

Compara com baseline BTC. Se ETH tiver edge similar/melhor:
- Adiciona ETH como segundo ativo em produção
- Multiplica número de trades sem prejudicar BTC

K incremental: +1.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from pipeline import features as feat, labels as lab  # noqa: E402

TIMEFRAME = 240
HORIZON = 12
ATR_MULT = 3.0
COST = 0.0015
BPD = 6
RETRAIN_EVERY = 90 * BPD
START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)
VAL_END = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
HOLDOUT_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
THR = 0.35
NO_BEAR = -0.05
BARS_PER_MONTH = 180
INITIAL = 1000.0

LGB = dict(objective="binary", metric="binary_logloss", verbose=-1, n_jobs=-1,
           learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, lambda_l2=0.5)
N_ROUNDS = 500


def build(asset: str):
    df = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME, lag=1, asset=asset).drop_nulls(subset=["atr_14"])
    lab_df = lab.triple_barrier(df, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=HORIZON)
    lab_df = lab.attach_uniqueness(lab_df, horizon_bars=HORIZON)
    lab_df = lab_df.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y"))
    excl = feat.LAG_SAFE_EXCLUDE | {"label","hit_bar","barrier_ret","upper_px","lower_px","y","uniqueness_weight"}
    fcols = [c for c in lab_df.columns if c not in excl]
    base = ["open_time","close","high","low","y","barrier_ret","uniqueness_weight","atr_14"]
    extra = [c for c in fcols if c not in base]
    m = lab_df.select(base + extra).drop_nulls(subset=fcols + ["y"]).to_pandas()
    m["dt"] = m["open_time"].apply(lambda ms: datetime.fromtimestamp(ms/1000, tz=timezone.utc))
    return m, fcols


def walk_forward(mat, fcols):
    n = len(mat); proba = np.zeros(n); covered = np.zeros(n, dtype=bool)
    pos = mat.index[mat["dt"] >= START_DATE].tolist()[0]
    while pos < n:
        train_end = pos - HORIZON
        if train_end < 500:
            pos += RETRAIN_EVERY
            continue
        X_tr = mat.iloc[:train_end][fcols].values
        y_tr = mat.iloc[:train_end]["y"].values
        w_tr = mat.iloc[:train_end]["uniqueness_weight"].values
        model = lgb.train(LGB, lgb.Dataset(X_tr, y_tr, weight=w_tr), num_boost_round=N_ROUNDS)
        block_end = min(pos + RETRAIN_EVERY, n)
        X_te = mat.iloc[pos:block_end][fcols].values
        proba[pos:block_end] = model.predict(X_te)
        covered[pos:block_end] = True
        pos = block_end
    return proba, covered


def simulate(mat, proba, covered):
    capital = INITIAL; cash = capital; position = None; trades = []; equity = []
    close = mat["close"].values; high = mat["high"].values; low = mat["low"].values
    ot = mat["open_time"].values; atr = mat["atr_14"].values
    for i in range(len(mat)):
        if not covered[i]:
            equity.append(capital); continue
        if position is not None:
            hit_stop = low[i] <= position["stop"]
            hit_target = high[i] >= position["target"]
            timeout = ot[i] >= position["timeout_at"]
            exit_p = None
            if hit_stop: exit_p = position["stop"]
            elif hit_target: exit_p = position["target"]
            elif timeout: exit_p = close[i]
            if exit_p is not None:
                pnl = (exit_p / position["entry"] - 1) - COST
                cash += position["size_usd"] * (1 + pnl)
                capital = cash
                trades.append({"pnl": pnl, "dt": mat["dt"].iloc[i]})
                position = None
        if position is None and proba[i] > THR:
            if i >= BARS_PER_MONTH:
                ret_30d = close[i] / close[i - BARS_PER_MONTH] - 1
                if ret_30d < NO_BEAR:
                    equity.append(capital); continue
            entry = close[i]
            stop = entry - ATR_MULT * float(atr[i])
            target = entry + ATR_MULT * float(atr[i])
            size_usd = capital
            cash -= size_usd
            position = {"entry": entry, "stop": stop, "target": target,
                        "size_usd": size_usd,
                        "timeout_at": ot[i] + HORIZON * 4 * 3600 * 1000}
        if position is not None:
            mtm = position["size_usd"] * (close[i] / position["entry"])
            equity.append(cash + mtm)
        else:
            equity.append(capital)
    return np.array(equity), pd.DataFrame(trades)


def report(eq, trades, mat, label):
    val_mask = ((mat["dt"] >= START_DATE) & (mat["dt"] <= VAL_END)).values
    ho_mask = (mat["dt"] >= HOLDOUT_START).values
    def sr(mask):
        r = pd.Series(eq[mask]).pct_change().fillna(0)
        return float((r.mean() / r.std()) * np.sqrt(BPD * 365)) if r.std() > 0 else 0
    peak = pd.Series(eq).cummax()
    dd = float((eq / peak - 1).min())
    win = float((trades["pnl"] > 0).mean()) if len(trades) else 0
    bh_entry = mat[(mat["dt"] >= START_DATE)]["close"].iloc[0]
    bh_final = mat["close"].iloc[-1]
    bh_cap = INITIAL * (bh_final / bh_entry)
    return dict(label=label, final=float(eq[-1]), sval=sr(val_mask),
                sho=sr(ho_mask), dd=dd, n=len(trades), win=win, bh=float(bh_cap))


for asset in ["BTC", "ETH"]:
    print(f"\n[{asset}] build matriz…", flush=True)
    t0 = time.time()
    mat, fcols = build(asset)
    print(f"  {len(mat)} rows, {len(fcols)} features  ({time.time()-t0:.0f}s)")
    print(f"[{asset}] walk-forward…", flush=True)
    t0 = time.time()
    proba, cov = walk_forward(mat, fcols)
    eq, tr = simulate(mat, proba, cov)
    r = report(eq, tr, mat, asset)
    print(f"  Final ${r['final']:>6,.0f}  VAL {r['sval']:+.2f}  HO {r['sho']:+.2f}  "
          f"trades={r['n']}  win={100*r['win']:.1f}%  MaxDD={100*r['dd']:+.1f}%  "
          f"B&H {asset}: ${r['bh']:,.0f}  ({time.time()-t0:.0f}s)")
