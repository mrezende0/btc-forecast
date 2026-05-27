# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # EXP — Sample weights por uniqueness (López de Prado, AFML cap.4)
#
# Triple-barrier com horizonte longo (12, 18 bars) gera labels que se sobrepõem:
# label i ainda está "vivo" enquanto i+1, i+2 começam. Treinar com peso uniforme
# conta a mesma informação várias vezes → otimismo. AFML eq.4.2 corrige isso
# com sample_weight = avg_uniqueness ∈ (0,1].
#
# Setup:
#   - dual-horizon: mid (h=12, 48h) + long (h=18, 72h)
#   - sinal: AND com threshold 0.35 nos 2 modelos
#   - walk-forward expanding quarterly 2023+
#   - cost=0.0015
#
# Comparação: baseline (sem weight) vs com uniqueness.

# %%
from __future__ import annotations
from pathlib import Path
import sys, os
ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import time
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from datetime import datetime, timezone

from pipeline import features as feat, labels as lab

TIMEFRAME = 240          # 4h
ATR_MULT = 3.0
COST = 0.0015
THRESHOLD = 0.35

HORIZONS = {"mid": 12, "long": 18}

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

# %% [markdown]
# ## 1. Build matrix em 4h

# %%
print("Building feature matrix @ 4h...")
df_base = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME, lag=1)
df_base = df_base.drop_nulls(subset=["atr_14"])
print(f"shape: {df_base.shape}")
print(
    f"range: "
    f"{datetime.fromtimestamp(df_base['open_time'].min()/1000, tz=timezone.utc).date()} → "
    f"{datetime.fromtimestamp(df_base['open_time'].max()/1000, tz=timezone.utc).date()}"
)

# %% [markdown]
# ## 2. Triple-barrier + uniqueness por horizonte

# %%
labels_by_h: dict[str, pl.DataFrame] = {}
for name, h in HORIZONS.items():
    print(f"\n--- Labeling horizonte {name} (h={h} bars / {h*4}h) ---")
    lbl = lab.triple_barrier(
        df_base, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=h
    )
    lbl = lab.attach_uniqueness(lbl, horizon_bars=h)
    lbl = lbl.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y_bin"))
    total = lbl.height
    for v, lab_name in [(1, "LONG_WIN"), (0, "TIMEOUT"), (-1, "STOP")]:
        n = lbl.filter(pl.col("label") == v).height
        print(f"  {lab_name:>8s}  {n:>5d}  ({100*n/total:>4.1f}%)")
    w = lbl["uniqueness_weight"].to_numpy()
    print(
        f"  uniqueness_weight  min={w.min():.3f}  p25={np.percentile(w,25):.3f}  "
        f"median={np.median(w):.3f}  p75={np.percentile(w,75):.3f}  max={w.max():.3f}  "
        f"mean={w.mean():.3f}"
    )
    labels_by_h[name] = lbl

# %% [markdown]
# ## 3. Features

# %%
sample = labels_by_h["mid"]
feature_cols = [
    c for c in sample.columns
    if c not in feat.LAG_SAFE_EXCLUDE
    and c not in {
        "label", "hit_bar", "barrier_ret", "upper_px", "lower_px",
        "y_bin", "uniqueness_weight",
    }
]
print(f"\nFeatures: {len(feature_cols)}")

mats: dict[str, pd.DataFrame] = {}
for name, lbl in labels_by_h.items():
    m = lbl.select([
        "open_time", "close", "y_bin", "barrier_ret",
        "uniqueness_weight", *feature_cols,
    ]).drop_nulls(subset=feature_cols + ["y_bin"]).to_pandas()
    m["dt"] = m["open_time"].apply(lambda ms: datetime.fromtimestamp(ms/1000, tz=timezone.utc))
    m["quarter"] = m["dt"].dt.to_period("Q")
    mats[name] = m
    print(f"  {name:>5s}: {len(m)} rows")

# %% [markdown]
# ## 4. Walk-forward — roda 2x (baseline e com_weight) para cada horizonte

# %%
quarters = [q for q in sorted(mats["mid"]["quarter"].unique()) if q.start_time.year >= 2023]
print(f"\nQuarters de teste: {len(quarters)} ({quarters[0]} → {quarters[-1]})")


def run_walkforward(use_weight: bool) -> dict[str, dict[int, float]]:
    """Treina dual-horizon e devolve dict {horizon_name: {open_time: proba}}."""
    proba_by_h: dict[str, dict[int, float]] = {h: {} for h in HORIZONS}
    tag = "with_w" if use_weight else "baseline"
    print(f"\n>>> Walk-forward [{tag}] (sample_weight={'uniqueness' if use_weight else 'uniform'})")
    for q in quarters:
        log_parts = [str(q)]
        for name, h in HORIZONS.items():
            m = mats[name]
            test_mask = m["quarter"] == q
            test_idx = m.index[test_mask].tolist()
            if not test_idx:
                continue
            test_start = test_idx[0]
            train_end = test_start - h
            test_use_start = test_start + h
            if train_end < 500 or test_use_start >= test_idx[-1]:
                continue
            train_idx = list(range(0, train_end))
            test_use_idx = [i for i in test_idx if i >= test_use_start]

            X_tr = m.iloc[train_idx][feature_cols].values
            y_tr = m.iloc[train_idx]["y_bin"].values
            X_te = m.iloc[test_use_idx][feature_cols].values
            ot_te = m.iloc[test_use_idx]["open_time"].values

            if use_weight:
                w_tr = m.iloc[train_idx]["uniqueness_weight"].values
                dtr = lgb.Dataset(X_tr, y_tr, weight=w_tr)
            else:
                dtr = lgb.Dataset(X_tr, y_tr)
            t0 = time.time()
            model = lgb.train(PARAMS, dtr, num_boost_round=N_ROUNDS)
            dt_train = time.time() - t0
            proba = model.predict(X_te)
            for ot, p in zip(ot_te, proba):
                proba_by_h[name][int(ot)] = float(p)
            log_parts.append(f"{name} n_te={len(test_use_idx)} ({dt_train:.1f}s)")
        print("  " + " | ".join(log_parts))
    return proba_by_h


proba_baseline = run_walkforward(use_weight=False)
proba_uniq = run_walkforward(use_weight=True)


# %% [markdown]
# ## 5. Avaliação dual-horizon AND

# %%
def sharpe(pnls: np.ndarray, trades_per_year: float) -> float:
    if len(pnls) < 2 or pnls.std(ddof=1) == 0:
        return 0.0
    return float(pnls.mean() / pnls.std(ddof=1) * np.sqrt(trades_per_year))


def max_drawdown(pnls_ordered: np.ndarray) -> float:
    """MaxDD da curva cumulativa de PnL (em pp da equity inicial = 1.0)."""
    if len(pnls_ordered) == 0:
        return 0.0
    eq = 1.0 + np.cumsum(pnls_ordered)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(dd.min())


def eval_dual(proba_by_h: dict[str, dict[int, float]], tag: str) -> dict:
    ots_common = sorted(set(proba_by_h["mid"].keys()) & set(proba_by_h["long"].keys()))
    p_mid = np.array([proba_by_h["mid"][o] for o in ots_common])
    p_long = np.array([proba_by_h["long"][o] for o in ots_common])

    # PnL com ret do horizonte mid (igual baseline v2)
    ret_lut_mid = dict(zip(mats["mid"]["open_time"].astype(int), mats["mid"]["barrier_ret"]))
    ret_ref = np.array([ret_lut_mid[int(o)] for o in ots_common])

    take = (p_mid > THRESHOLD) & (p_long > THRESHOLD)
    n_sig = int(take.sum())
    # ordenar trades por open_time para curva equity
    ots_arr = np.array(ots_common)
    take_idx_sorted = np.argsort(ots_arr[take])
    pnls = ret_ref[take] - COST
    pnls_ordered = pnls[take_idx_sorted]

    n_years = max(1e-9, (ots_common[-1] - ots_common[0]) / 1000 / 86400 / 365)
    trades_per_year = max(1.0, n_sig / n_years) if n_sig else 1.0
    sh = sharpe(pnls, trades_per_year) if n_sig else 0.0
    win = float((pnls > 0).mean()) if n_sig else 0.0
    avg = float(pnls.mean()) if n_sig else 0.0
    tot = float(pnls.sum()) if n_sig else 0.0
    gross_gains = pnls[pnls > 0].sum() if n_sig else 0.0
    gross_loss = -pnls[pnls < 0].sum() if n_sig else 0.0
    pf = float(gross_gains / gross_loss) if gross_loss > 0 else float("inf")
    mdd = max_drawdown(pnls_ordered)

    print(f"\n=== Dual-horizon AND [{tag}] (pool {n_years:.2f}y, {len(ots_common)} bars) ===")
    print(f"  n_signals     = {n_sig}")
    print(f"  win_rate      = {100*win:.1f}%")
    print(f"  avg_pnl       = {100*avg:+.3f}%")
    print(f"  tot_pnl       = {100*tot:+.2f}%")
    print(f"  sharpe        = {sh:+.2f}")
    print(f"  profit_factor = {pf:.2f}")
    print(f"  max_drawdown  = {100*mdd:.2f}%")
    return dict(
        n_sig=n_sig, win=win, avg=avg, tot=tot, sharpe=sh, pf=pf, mdd=mdd,
        n_years=n_years,
    )


res_base = eval_dual(proba_baseline, "baseline (no weight)")
res_uniq = eval_dual(proba_uniq, "with uniqueness")

# %% [markdown]
# ## 6. Resumo + recomendação

# %%
print("\n" + "=" * 78)
print("RESUMO — Dual-horizon AND, threshold 0.35, cost=0.0015")
print("=" * 78)
hdr = f"{'metric':<14s} {'baseline':>12s} {'uniqueness':>12s} {'delta':>12s}"
print(hdr)
print("-" * len(hdr))
rows = [
    ("sharpe",      res_base["sharpe"],   res_uniq["sharpe"],   res_uniq["sharpe"] - res_base["sharpe"]),
    ("tot_pnl_%",   100*res_base["tot"],  100*res_uniq["tot"],  100*(res_uniq["tot"] - res_base["tot"])),
    ("avg_pnl_%",   100*res_base["avg"],  100*res_uniq["avg"],  100*(res_uniq["avg"] - res_base["avg"])),
    ("win_rate_%",  100*res_base["win"],  100*res_uniq["win"],  100*(res_uniq["win"] - res_base["win"])),
    ("profit_fact", res_base["pf"],       res_uniq["pf"],       res_uniq["pf"] - res_base["pf"]),
    ("max_dd_%",    100*res_base["mdd"],  100*res_uniq["mdd"],  100*(res_uniq["mdd"] - res_base["mdd"])),
    ("n_signals",   res_base["n_sig"],    res_uniq["n_sig"],    res_uniq["n_sig"] - res_base["n_sig"]),
]
for name, a, b, d in rows:
    print(f"{name:<14s} {a:>12.3f} {b:>12.3f} {d:>+12.3f}")

print("\nDistribuição uniqueness_weight (mid h=12):")
w_mid = labels_by_h["mid"]["uniqueness_weight"].to_numpy()
print(
    f"  min={w_mid.min():.3f}  p25={np.percentile(w_mid,25):.3f}  "
    f"median={np.median(w_mid):.3f}  p75={np.percentile(w_mid,75):.3f}  "
    f"max={w_mid.max():.3f}  mean={w_mid.mean():.3f}"
)
print("Distribuição uniqueness_weight (long h=18):")
w_long = labels_by_h["long"]["uniqueness_weight"].to_numpy()
print(
    f"  min={w_long.min():.3f}  p25={np.percentile(w_long,25):.3f}  "
    f"median={np.median(w_long):.3f}  p75={np.percentile(w_long,75):.3f}  "
    f"max={w_long.max():.3f}  mean={w_long.mean():.3f}"
)

delta_sh = res_uniq["sharpe"] - res_base["sharpe"]
delta_tot = res_uniq["tot"] - res_base["tot"]
print("\nVeredito:")
if delta_sh >= 0.10 and delta_tot > 0:
    veredito = "INTEGRAR — ganho material de Sharpe."
elif delta_sh > 0 and delta_tot > 0:
    veredito = "MARGINAL — ganho pequeno; considerar antes de produção."
elif abs(delta_sh) < 0.05 and abs(delta_tot) < 0.02:
    veredito = "NEUTRO — peso não muda nada material; descartar por simplicidade."
else:
    veredito = "DESCARTAR — uniqueness piora a estratégia."
print(f"  ΔSharpe={delta_sh:+.2f}  ΔTotPnL={100*delta_tot:+.2f}%  →  {veredito}")
