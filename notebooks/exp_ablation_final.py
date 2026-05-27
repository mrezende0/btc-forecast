"""exp_ablation_final — Ablação científica das melhorias do modelo.

Compara 6 configurações isoladamente pra entender o que cada melhoria entrega:
  A) MINIMAL          — só LGB + threshold 0.35, COST 0.0015, FULL sizing
  B) +UNIQUENESS      — A + sample weights por uniqueness (LdP)
  C) +NO_BEAR         — B + filtro sem-BEAR (-5% no mês)  ← PRODUÇÃO ATUAL
  D) +DUAL_AND        — C + dual-horizon AND (legado, removido)
  E) +SENTIMENT       — C + sentiment GDELT/FinBERT features
  F) +FLOW            — C + flow features (taker_buy, basis, OFI)

Cada uma: $1000 inicial, 4h bars, 1 posição por vez, compounding sequencial.
Reporta Sharpe VAL/HOLDOUT, final $, win rate, MaxDD, trades.

K incremental: +1 (1 ablação como bloco).
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
H_MID = 12
H_LONG = 18
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


def build(horizon: int = H_MID):
    df = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME, lag=1).drop_nulls(subset=["atr_14"])
    lab_df = lab.triple_barrier(df, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=horizon)
    lab_df = lab.attach_uniqueness(lab_df, horizon_bars=horizon)
    lab_df = lab_df.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y"))
    excl = feat.LAG_SAFE_EXCLUDE | {"label","hit_bar","barrier_ret","upper_px","lower_px","y","uniqueness_weight"}
    fcols = [c for c in lab_df.columns if c not in excl]
    base = ["open_time","close","high","low","y","barrier_ret","uniqueness_weight","atr_14"]
    extra = [c for c in fcols if c not in base]
    m = lab_df.select(base + extra).drop_nulls(subset=fcols + ["y"]).to_pandas()
    m["dt"] = m["open_time"].apply(lambda ms: datetime.fromtimestamp(ms/1000, tz=timezone.utc))
    return m, fcols


def walk_forward(mat, fcols, horizon=H_MID, use_uniqueness=False):
    n = len(mat); proba = np.zeros(n); covered = np.zeros(n, dtype=bool)
    pos = mat.index[mat["dt"] >= START_DATE].tolist()[0]
    while pos < n:
        train_end = pos - horizon
        if train_end < 500:
            pos += RETRAIN_EVERY
            continue
        X_tr = mat.iloc[:train_end][fcols].values
        y_tr = mat.iloc[:train_end]["y"].values
        w = mat.iloc[:train_end]["uniqueness_weight"].values if use_uniqueness else None
        ds = lgb.Dataset(X_tr, y_tr, weight=w)
        model = lgb.train(LGB, ds, num_boost_round=N_ROUNDS)
        block_end = min(pos + RETRAIN_EVERY, n)
        X_te = mat.iloc[pos:block_end][fcols].values
        proba[pos:block_end] = model.predict(X_te)
        covered[pos:block_end] = True
        pos = block_end
    return proba, covered


def simulate(mat, proba_mid, covered_mid, proba_long=None, covered_long=None,
             use_no_bear=False, use_and_rule=False):
    capital = INITIAL; cash = capital; position = None; trades = []; equity = []
    close = mat["close"].values; high = mat["high"].values; low = mat["low"].values
    ot = mat["open_time"].values; atr = mat["atr_14"].values

    for i in range(len(mat)):
        cov = covered_mid[i] and (covered_long[i] if covered_long is not None else True)
        if not cov:
            equity.append(capital); continue
        # close existing
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
                trades.append({"pnl": pnl})
                position = None
        # open new
        if position is None:
            sig_mid = proba_mid[i] > THR
            sig_long = (proba_long is not None) and (proba_long[i] > THR)
            if use_and_rule:
                sig = sig_mid and sig_long
            else:
                sig = sig_mid
            if sig:
                if use_no_bear and i >= BARS_PER_MONTH:
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
                            "timeout_at": ot[i] + H_MID * 4 * 3600 * 1000}
        if position is not None:
            mtm = position["size_usd"] * (close[i] / position["entry"])
            equity.append(cash + mtm)
        else:
            equity.append(capital)
    return np.array(equity), pd.DataFrame(trades)


def report(eq, trades, mat, label):
    val_mask = ((mat["dt"] >= START_DATE) & (mat["dt"] <= VAL_END)).values
    ho_mask = (mat["dt"] >= HOLDOUT_START).values
    def sharpe(mask):
        r = pd.Series(eq[mask]).pct_change().fillna(0)
        return float((r.mean() / r.std()) * np.sqrt(BPD * 365)) if r.std() > 0 else 0
    peak = pd.Series(eq).cummax()
    dd = float((eq / peak - 1).min())
    win = float((trades["pnl"] > 0).mean()) if len(trades) else 0
    return dict(label=label, final=float(eq[-1]), sval=sharpe(val_mask),
                sho=sharpe(ho_mask), dd=dd, n=len(trades), win=win)


# ========================= MAIN =========================
print("[abl] build matrizes (mid + long)…", flush=True)
m_mid, fc_mid = build(H_MID)
m_long, fc_long = build(H_LONG)

# Identifica grupos de features
sent_cols = [c for c in fc_mid if any(t in c for t in ["sentiment", "news_count"])]
flow_cols = [c for c in fc_mid if any(t in c for t in ["taker_buy", "ofi_proxy", "basis", "flow_div", "perp_taker"])]
base_cols = [c for c in fc_mid if c not in sent_cols and c not in flow_cols]
print(f"  base={len(base_cols)} sent={len(sent_cols)} flow={len(flow_cols)}")

results = []

# A) MINIMAL
print("\n[A] MINIMAL (sem uniqueness, sem no_bear, base+sent+flow)…", flush=True)
t0 = time.time()
proba, cov = walk_forward(m_mid, fc_mid, use_uniqueness=False)
eq, tr = simulate(m_mid, proba, cov, use_no_bear=False)
results.append(report(eq, tr, m_mid, "A) MINIMAL"))
print(f"  ({time.time()-t0:.0f}s)  {results[-1]}")

# B) + UNIQUENESS
print("\n[B] +UNIQUENESS (LdP sample weights)…", flush=True)
t0 = time.time()
proba, cov = walk_forward(m_mid, fc_mid, use_uniqueness=True)
eq, tr = simulate(m_mid, proba, cov, use_no_bear=False)
results.append(report(eq, tr, m_mid, "B) +UNIQUENESS"))
print(f"  ({time.time()-t0:.0f}s)  {results[-1]}")

# C) + NO_BEAR (= PRODUÇÃO ATUAL)
print("\n[C] +NO_BEAR (★ PRODUÇÃO ATUAL)…", flush=True)
eq, tr = simulate(m_mid, proba, cov, use_no_bear=True)
results.append(report(eq, tr, m_mid, "C) ★ PRODUÇÃO"))
print(f"  {results[-1]}")

# D) + DUAL_AND (legado removido)
print("\n[D] +DUAL_AND (mid + long AND, legado)…", flush=True)
t0 = time.time()
proba_long, cov_long = walk_forward(m_long, fc_long, horizon=H_LONG, use_uniqueness=True)
# Map proba_long pra grid mid via open_time
ot_long_idx = {ot: i for i, ot in enumerate(m_long["open_time"].values)}
proba_long_aligned = np.zeros(len(m_mid))
cov_long_aligned = np.zeros(len(m_mid), dtype=bool)
for j, ot_val in enumerate(m_mid["open_time"].values):
    k = ot_long_idx.get(ot_val)
    if k is not None:
        proba_long_aligned[j] = proba_long[k]
        cov_long_aligned[j] = cov_long[k]
eq, tr = simulate(m_mid, proba, cov, proba_long=proba_long_aligned,
                  covered_long=cov_long_aligned, use_no_bear=True, use_and_rule=True)
results.append(report(eq, tr, m_mid, "D) +DUAL_AND"))
print(f"  ({time.time()-t0:.0f}s)  {results[-1]}")

# E) + SENTIMENT FEATURES (no modelo mid)
print("\n[E] +SENTIMENT (features no LGB)…", flush=True)
t0 = time.time()
fc_with_sent = base_cols + sent_cols  # remove flow se houver
proba_e, cov_e = walk_forward(m_mid, fc_with_sent, use_uniqueness=True)
eq, tr = simulate(m_mid, proba_e, cov_e, use_no_bear=True)
results.append(report(eq, tr, m_mid, "E) +SENTIMENT (features)"))
print(f"  ({time.time()-t0:.0f}s)  {results[-1]}")

# F) + FLOW FEATURES
print("\n[F] +FLOW (taker + basis + OFI no LGB)…", flush=True)
t0 = time.time()
fc_with_flow = base_cols + flow_cols  # base sem sent
proba_f, cov_f = walk_forward(m_mid, fc_with_flow, use_uniqueness=True)
eq, tr = simulate(m_mid, proba_f, cov_f, use_no_bear=True)
results.append(report(eq, tr, m_mid, "F) +FLOW (features)"))
print(f"  ({time.time()-t0:.0f}s)  {results[-1]}")

# === TABELA FINAL ===
print("\n" + "=" * 100)
print(f"{'Config':<28s}  {'Final':>10s}  {'Retorno':>9s}  {'VAL Shr':>8s}  {'HO Shr':>7s}  {'MaxDD':>7s}  {'Trades':>7s}  {'Win%':>5s}")
print("-" * 100)
for r in results:
    ret = (r["final"] / INITIAL - 1) * 100
    print(f"  {r['label']:<26s}  ${r['final']:>8,.0f}  {ret:>+7.1f}%  {r['sval']:>+7.2f}  {r['sho']:>+7.2f}  {100*r['dd']:>+6.1f}%  {r['n']:>7d}  {100*r['win']:>4.1f}%")
print("=" * 100)

# B&H mesmo período
first_idx = m_mid.index[m_mid["dt"] >= START_DATE].tolist()[0]
bh_entry = float(m_mid["close"].iloc[first_idx])
bh_final = float(m_mid["close"].iloc[-1])
bh_eq = INITIAL * (m_mid["close"].iloc[first_idx:] / bh_entry).values
bh_peak = pd.Series(bh_eq).cummax()
bh_dd = float((bh_eq / bh_peak - 1).min())
bh_r = pd.Series(bh_eq).pct_change().fillna(0)
bh_sharpe = float((bh_r.mean() / bh_r.std()) * np.sqrt(BPD * 365)) if bh_r.std() > 0 else 0
ret_bh = bh_final / bh_entry - 1
print(f"  {'Buy-and-Hold':<26s}  ${bh_eq[-1]:>8,.0f}  {100*ret_bh:>+7.1f}%  {'n/a':>7s}  {bh_sharpe:>+7.2f}  {100*bh_dd:>+6.1f}%  {1:>7d}  {'n/a':>5s}")
print("=" * 100)
