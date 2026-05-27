"""exp_a2_flow_features — A2 do roadmap_v2.

Testa se features de fluxo (taker_buy_ratio, OFI proxy, basis spot-perp,
flow divergence) melhoram o WINNER A1-A (MID-only, thr=0.35, no_bear=-0.05).

Comparação honesta:
  baseline: features atuais SEM flow
  flow:     features atuais + flow (taker + basis)

Walk-forward com retreino quarterly, COST=0.0015, uniqueness weighting,
simulação realista (1 posição por vez, compounding).

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
BARS_PER_DAY = 6
RETRAIN_EVERY = 90 * BARS_PER_DAY
START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)
VAL_END = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
HOLDOUT_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
THR = 0.35
NO_BEAR = -0.05
BARS_PER_MONTH = 180
INITIAL_CAPITAL = 1000.0

LGB_PARAMS = dict(
    objective="binary", metric="binary_logloss", verbose=-1, n_jobs=-1,
    learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, lambda_l2=0.5,
)
N_ROUNDS = 500


def build_matrix():
    df = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME, lag=1).drop_nulls(subset=["atr_14"])
    lab_df = lab.triple_barrier(df, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=HORIZON)
    lab_df = lab.attach_uniqueness(lab_df, horizon_bars=HORIZON)
    lab_df = lab_df.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y"))
    excl = feat.LAG_SAFE_EXCLUDE | {"label","hit_bar","barrier_ret","upper_px","lower_px","y","uniqueness_weight"}
    all_fcols = [c for c in lab_df.columns if c not in excl]
    base = ["open_time","close","high","low","y","barrier_ret","uniqueness_weight","atr_14"]
    extra = [c for c in all_fcols if c not in base]
    m = lab_df.select(base + extra).drop_nulls(subset=all_fcols + ["y"]).to_pandas()
    m["dt"] = m["open_time"].apply(lambda ms: datetime.fromtimestamp(ms/1000, tz=timezone.utc))
    return m, all_fcols


def walk_forward(mat, fcols):
    n = len(mat)
    proba = np.zeros(n)
    covered = np.zeros(n, dtype=bool)
    start_idx = mat.index[mat["dt"] >= START_DATE].tolist()
    pos = start_idx[0]
    while pos < n:
        train_end = pos - HORIZON
        if train_end < 500:
            pos += RETRAIN_EVERY
            continue
        X_tr = mat.iloc[:train_end][fcols].values
        y_tr = mat.iloc[:train_end]["y"].values
        w_tr = mat.iloc[:train_end]["uniqueness_weight"].values
        model = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, y_tr, weight=w_tr), num_boost_round=N_ROUNDS)
        block_end = min(pos + RETRAIN_EVERY, n)
        X_te = mat.iloc[pos:block_end][fcols].values
        proba[pos:block_end] = model.predict(X_te)
        covered[pos:block_end] = True
        pos = block_end
    return proba, covered, model  # last model pra feature importance


def simulate(mat, proba, covered):
    capital = INITIAL_CAPITAL
    cash = capital
    position = None
    trades = []
    equity = []
    close = mat["close"].values
    high = mat["high"].values
    low = mat["low"].values
    open_time = mat["open_time"].values
    atr_arr = mat["atr_14"].values

    for i in range(len(mat)):
        if not covered[i]:
            equity.append(capital)
            continue
        if position is not None:
            hit_stop = low[i] <= position["stop"]
            hit_target = high[i] >= position["target"]
            timeout = open_time[i] >= position["timeout_at"]
            exit_p = None
            if hit_stop:
                exit_p = position["stop"]
            elif hit_target:
                exit_p = position["target"]
            elif timeout:
                exit_p = close[i]
            if exit_p is not None:
                pnl = (exit_p / position["entry"] - 1) - COST
                pnl_usd = position["size_usd"] * pnl
                cash += position["size_usd"] + pnl_usd
                capital = cash
                trades.append({"dt": mat["dt"].iloc[i], "pnl": pnl})
                position = None
        if position is None and proba[i] > THR:
            if i >= BARS_PER_MONTH:
                ret_30d = close[i] / close[i - BARS_PER_MONTH] - 1
                if ret_30d < NO_BEAR:
                    equity.append(capital)
                    continue
            atr = float(atr_arr[i])
            entry = close[i]
            stop = entry - ATR_MULT * atr
            target = entry + ATR_MULT * atr
            size_usd = capital
            cash -= size_usd
            position = {"entry": entry, "stop": stop, "target": target,
                        "size_usd": size_usd,
                        "timeout_at": open_time[i] + HORIZON * 4 * 3600 * 1000}
        if position is not None:
            mtm = position["size_usd"] * (close[i] / position["entry"])
            equity.append(cash + mtm)
        else:
            equity.append(capital)
    return np.array(equity), pd.DataFrame(trades)


def report(eq, trades, mat, label):
    val_mask = (mat["dt"] >= START_DATE) & (mat["dt"] <= VAL_END)
    ho_mask = mat["dt"] >= HOLDOUT_START
    def sr(mask):
        r = pd.Series(eq[mask.values]).pct_change().fillna(0)
        return float((r.mean() / r.std()) * np.sqrt(BARS_PER_DAY * 365)) if r.std() > 0 else 0
    final = float(eq[-1])
    s_val = sr(val_mask); s_ho = sr(ho_mask)
    print(f"  {label:<25s}  Final ${final:>6,.0f}  VAL Sharpe={s_val:+.2f}  HO Sharpe={s_ho:+.2f}  Trades={len(trades)}")
    return {"label": label, "final": final, "val_sharpe": s_val, "ho_sharpe": s_ho, "n": len(trades)}


# ====================== MAIN ======================
print("[a2] build matriz com TODAS features (incl flow)…", flush=True)
mat, all_fcols = build_matrix()
print(f"[a2] {len(mat)} rows, {len(all_fcols)} features totais")

flow_cols = [c for c in all_fcols if any(t in c for t in ["taker_buy", "ofi", "basis", "flow_div", "perp_taker"])]
base_cols = [c for c in all_fcols if c not in flow_cols]
print(f"[a2] base: {len(base_cols)} features  |  flow: {len(flow_cols)}: {flow_cols}")

print("\n=== TESTE 1: BASELINE (sem flow) ===", flush=True)
t0 = time.time()
proba_b, cov_b, _ = walk_forward(mat, base_cols)
eq_b, trades_b = simulate(mat, proba_b, cov_b)
r1 = report(eq_b, trades_b, mat, "BASELINE")
print(f"  ({time.time()-t0:.0f}s)")

print("\n=== TESTE 2: BASE + FLOW ===", flush=True)
t0 = time.time()
proba_f, cov_f, model_f = walk_forward(mat, all_fcols)
eq_f, trades_f = simulate(mat, proba_f, cov_f)
r2 = report(eq_f, trades_f, mat, "BASE + FLOW")
print(f"  ({time.time()-t0:.0f}s)")

# Feature importance último fold do FLOW
print("\n=== Top 15 features (último fold, BASE+FLOW) ===")
imp = pd.DataFrame({
    "feature": all_fcols,
    "gain": model_f.feature_importance(importance_type="gain"),
}).sort_values("gain", ascending=False)
print(imp.head(15).to_string(index=False))

print("\nRanking flow features:")
for fc in flow_cols:
    if fc in imp["feature"].values:
        rank = list(imp["feature"]).index(fc) + 1
        gain = imp[imp["feature"] == fc]["gain"].iloc[0]
        print(f"  {fc:<28s}  rank #{rank:>2d}  gain={gain:.0f}")

# Veredito
print("\n" + "=" * 70)
print(f"BASELINE:   Final ${r1['final']:,.0f}  VAL {r1['val_sharpe']:+.2f}  HO {r1['ho_sharpe']:+.2f}  trades {r1['n']}")
print(f"BASE+FLOW:  Final ${r2['final']:,.0f}  VAL {r2['val_sharpe']:+.2f}  HO {r2['ho_sharpe']:+.2f}  trades {r2['n']}")
print(f"DELTA HO Sharpe: {r2['ho_sharpe'] - r1['ho_sharpe']:+.2f}")
print(f"DELTA Final $:   {r2['final'] - r1['final']:+.0f}")
if r2["ho_sharpe"] > r1["ho_sharpe"] + 0.10:
    print("VEREDITO: ✅ INTEGRAR flow features (delta > 0.10 Sharpe)")
elif r2["ho_sharpe"] < r1["ho_sharpe"] - 0.05:
    print("VEREDITO: ❌ DESCARTAR (flow piorou Sharpe)")
else:
    print("VEREDITO: ⚪ MARGINAL — não vale a complexidade")
