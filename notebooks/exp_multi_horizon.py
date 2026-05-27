# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # EXP — Multi-horizon ensemble
#
# Treina 3 modelos LightGBM binary independentes com horizontes distintos:
#   - h_short = 6 bars (24h)
#   - h_mid   = 12 bars (48h)  ← baseline v2
#   - h_long  = 18 bars (72h)
#
# Cada modelo tem seu próprio triple-barrier (±3×ATR), seu próprio purge no
# walk-forward (= horizon_bars) e seu próprio threshold 0.35.
#
# Combina sinais via voto em 3 regras:
#   - consensus: TODOS os 3 modelos disseram LONG
#   - majority:  >=2 dos 3
#   - any:       >=1 dos 3
#
# PnL é calculado SEMPRE com `ret_te` do horizonte mid (48h) — fair comparison
# vs baseline (esse é o retorno realizado no mesmo trade que o baseline tomaria).

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
COST = 0.0015  # Binance taker 0.10% × 2 + slippage real
THRESHOLD = 0.35
BARS_PER_YEAR = 6 * 365  # 4h => 6 bars/day

HORIZONS = {"short": 6, "mid": 12, "long": 18}

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
# ## 1. Build matrix em 4h (uma vez só)

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
# ## 2. Gera 3 sets de labels (um por horizonte)

# %%
labels_by_h: dict[str, pl.DataFrame] = {}
for name, h in HORIZONS.items():
    print(f"\n--- Labeling horizonte {name} (h={h} bars / {h*4}h) ---")
    lbl = lab.triple_barrier(
        df_base, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=h
    )
    total = lbl.height
    for v, lab_name in [(1, "LONG_WIN"), (0, "TIMEOUT"), (-1, "STOP")]:
        n = lbl.filter(pl.col("label") == v).height
        print(f"  {lab_name:>8s}  {n:>5d}  ({100*n/total:>4.1f}%)")
    lbl = lbl.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y_bin"))
    print(f"  base_rate LONG_WIN: {100*lbl['y_bin'].mean():.1f}%")
    labels_by_h[name] = lbl

# %% [markdown]
# ## 3. Define feature set (igual para os 3)

# %%
sample = labels_by_h["mid"]
feature_cols = [
    c for c in sample.columns
    if c not in feat.LAG_SAFE_EXCLUDE
    and c not in {"label", "hit_bar", "barrier_ret", "upper_px", "lower_px", "y_bin"}
]
print(f"\nFeatures: {len(feature_cols)}")

# %% [markdown]
# ## 4. Constrói matrizes pandas — uma por horizonte
#    Importante: cada matriz tem seu próprio drop_nulls em barrier_ret;
#    o que importa para o ensemble é alinhar pelo open_time.

# %%
mats: dict[str, pd.DataFrame] = {}
for name, lbl in labels_by_h.items():
    m = lbl.select(["open_time", "close", "y_bin", "barrier_ret", *feature_cols]).drop_nulls(
        subset=feature_cols + ["y_bin"]
    ).to_pandas()
    m["dt"] = m["open_time"].apply(lambda ms: datetime.fromtimestamp(ms/1000, tz=timezone.utc))
    m["quarter"] = m["dt"].dt.to_period("Q")
    mats[name] = m
    print(f"  {name:>5s}: {len(m)} rows, range {m['dt'].min().date()} → {m['dt'].max().date()}")

# Garante que open_time é alinhável (mesmo grid 4h). Para PnL "fair", usamos
# ret do horizonte mid; juntaremos por open_time depois.

# %% [markdown]
# ## 5. Walk-forward — roda os 3 modelos no mesmo loop quarterly

# %%
# Define grade de quarters comum (>=2023) a partir do mid (referência)
quarters = [q for q in sorted(mats["mid"]["quarter"].unique()) if q.start_time.year >= 2023]
print(f"\nQuarters de teste: {len(quarters)} ({quarters[0]} → {quarters[-1]})")

# Acumuladores por horizonte: open_time -> proba
proba_by_h: dict[str, dict[int, float]] = {h: {} for h in HORIZONS}
# Para baseline standalone (PnL real de cada h)
quarter_results: list[dict] = []

for q in quarters:
    qrow = {"quarter": str(q)}
    last_dt = None
    for name, h in HORIZONS.items():
        m = mats[name]
        test_mask = m["quarter"] == q
        test_idx = m.index[test_mask].tolist()
        if not test_idx:
            qrow[f"{name}_n_sig"] = 0
            continue
        test_start = test_idx[0]
        # Purge específico desse horizonte
        train_end = test_start - h
        test_use_start = test_start + h
        if train_end < 500 or test_use_start >= test_idx[-1]:
            qrow[f"{name}_n_sig"] = 0
            continue
        train_idx = list(range(0, train_end))
        test_use_idx = [i for i in test_idx if i >= test_use_start]

        X_tr = m.iloc[train_idx][feature_cols].values
        y_tr = m.iloc[train_idx]["y_bin"].values
        X_te = m.iloc[test_use_idx][feature_cols].values
        y_te = m.iloc[test_use_idx]["y_bin"].values
        ret_te = m.iloc[test_use_idx]["barrier_ret"].values
        ot_te = m.iloc[test_use_idx]["open_time"].values

        t0 = time.time()
        dtr = lgb.Dataset(X_tr, y_tr)
        model = lgb.train(PARAMS, dtr, num_boost_round=N_ROUNDS)
        dt_train = time.time() - t0
        proba = model.predict(X_te)

        # Persiste prob por open_time para uso no ensemble
        for ot, p in zip(ot_te, proba):
            proba_by_h[name][int(ot)] = float(p)

        # Métrica standalone desse horizonte (PnL é com ret_te DO PROPRIO horizonte)
        take = proba > THRESHOLD
        n_sig = int(take.sum())
        pnl = (ret_te[take] - COST) if n_sig else np.array([])
        avg_pnl = pnl.mean() if n_sig else 0.0
        tot_pnl = pnl.sum() if n_sig else 0.0
        qrow[f"{name}_n_sig"] = n_sig
        qrow[f"{name}_avg_pnl"] = avg_pnl
        qrow[f"{name}_tot_pnl"] = tot_pnl
        qrow[f"{name}_secs"] = dt_train
    quarter_results.append(qrow)
    print(
        f"{str(q):>8s}  "
        f"short n={qrow.get('short_n_sig',0):>3d} totPnL={100*qrow.get('short_tot_pnl',0):+5.2f}%  |  "
        f"mid n={qrow.get('mid_n_sig',0):>3d} totPnL={100*qrow.get('mid_tot_pnl',0):+5.2f}%  |  "
        f"long n={qrow.get('long_n_sig',0):>3d} totPnL={100*qrow.get('long_tot_pnl',0):+5.2f}%"
    )

# %% [markdown]
# ## 6. Sumário standalone por horizonte (baseline para cada)
#    PnL usa ret do próprio horizonte → reflete a estratégia "rodar só esse h".

# %%
def sharpe(pnls: np.ndarray, trades_per_year: float) -> float:
    if len(pnls) < 2 or pnls.std(ddof=1) == 0:
        return 0.0
    return float(pnls.mean() / pnls.std(ddof=1) * np.sqrt(trades_per_year))


print("\n" + "=" * 78)
print("STANDALONE — cada horizonte sozinho (PnL com seu próprio ret)")
print("=" * 78)
standalone_metrics = {}
for name, h in HORIZONS.items():
    m = mats[name]
    # Recolhe os trades realizados a partir das probas armazenadas
    rows = []
    for ot, p in proba_by_h[name].items():
        # ret correspondente
        # construir um lookup rápido por open_time -> ret
        pass
    # Lookup pelas chaves de proba
    lut = dict(zip(m["open_time"].astype(int), m["barrier_ret"]))
    probas = np.array(list(proba_by_h[name].values()))
    ots = np.array(list(proba_by_h[name].keys()))
    rets = np.array([lut[int(ot)] for ot in ots])
    take = probas > THRESHOLD
    n_sig = int(take.sum())
    pnls = rets[take] - COST
    # taxa anualização: trades_per_year = (n_sig / n_anos_test)
    n_years = max(1e-9, (max(ots) - min(ots)) / 1000 / 86400 / 365)
    trades_per_year = max(1.0, n_sig / n_years) if n_sig else 1.0
    # PnL bar de teste (densidade) — usamos diretamente media/std dos trades
    sh = sharpe(pnls, trades_per_year) if n_sig else 0.0
    avg = pnls.mean() if n_sig else 0.0
    win = float((pnls > 0).mean()) if n_sig else 0.0
    tot = pnls.sum() if n_sig else 0.0
    standalone_metrics[name] = {
        "n_sig": n_sig,
        "avg_pnl": avg,
        "tot_pnl": tot,
        "win_rate": win,
        "sharpe": sh,
        "trades_per_year": trades_per_year,
    }
    print(
        f"  {name:>5s} (h={h:>2d})  n_sig={n_sig:>4d}  win={100*win:>4.1f}%  "
        f"avg={100*avg:+.3f}%  tot={100*tot:+6.2f}%  sharpe={sh:+.2f}  "
        f"trades/y={trades_per_year:.1f}"
    )

# %% [markdown]
# ## 7. Ensemble por voto — PnL com ret_mid (fair comparison)

# %%
# Alinha por open_time: pega o set de OTs presentes em TODOS os 3 horizontes
ots_common = (
    set(proba_by_h["short"].keys())
    & set(proba_by_h["mid"].keys())
    & set(proba_by_h["long"].keys())
)
ots_common = sorted(ots_common)
print(f"\nOpen_times comuns aos 3 modelos: {len(ots_common)}")

# Vetores alinhados
p_short = np.array([proba_by_h["short"][o] for o in ots_common])
p_mid = np.array([proba_by_h["mid"][o] for o in ots_common])
p_long = np.array([proba_by_h["long"][o] for o in ots_common])

# Ret de referência = ret do horizonte mid (igual baseline)
ret_lut_mid = dict(zip(mats["mid"]["open_time"].astype(int), mats["mid"]["barrier_ret"]))
ret_ref = np.array([ret_lut_mid[int(o)] for o in ots_common])

s_short = p_short > THRESHOLD
s_mid = p_mid > THRESHOLD
s_long = p_long > THRESHOLD
votes = s_short.astype(int) + s_mid.astype(int) + s_long.astype(int)

# Anos do pool comum
n_years_pool = max(1e-9, (ots_common[-1] - ots_common[0]) / 1000 / 86400 / 365)

print("\n" + "=" * 78)
print(f"ENSEMBLE — PnL com ret do horizonte MID ({n_years_pool:.2f} anos no pool)")
print("=" * 78)

# Baseline (= só mid, no mesmo pool comum para comparabilidade)
baseline_take = s_mid
n_bl = int(baseline_take.sum())
pnls_bl = ret_ref[baseline_take] - COST
sh_bl = sharpe(pnls_bl, n_bl / n_years_pool) if n_bl else 0.0
print(
    f"  baseline (só mid)        n_sig={n_bl:>4d}  "
    f"win={100*(pnls_bl>0).mean():>4.1f}%  "
    f"avg={100*pnls_bl.mean():+.3f}%  tot={100*pnls_bl.sum():+6.2f}%  sharpe={sh_bl:+.2f}"
)

ensemble_results = {"baseline_mid": {"n_sig": n_bl, "sharpe": sh_bl, "tot": float(pnls_bl.sum())}}

for rule_name, mask in [
    ("consensus (3/3)", votes == 3),
    ("majority  (>=2)", votes >= 2),
    ("any       (>=1)", votes >= 1),
]:
    n = int(mask.sum())
    if n == 0:
        print(f"  {rule_name:>17s}        n_sig=   0")
        continue
    pnls = ret_ref[mask] - COST
    sh = sharpe(pnls, n / n_years_pool)
    print(
        f"  {rule_name:>17s}        n_sig={n:>4d}  "
        f"win={100*(pnls>0).mean():>4.1f}%  "
        f"avg={100*pnls.mean():+.3f}%  tot={100*pnls.sum():+6.2f}%  sharpe={sh:+.2f}"
    )
    ensemble_results[rule_name] = {"n_sig": n, "sharpe": sh, "tot": float(pnls.sum())}

# %% [markdown]
# ## 8. Recomendação final

# %%
print("\n" + "=" * 78)
print("RESUMO FINAL")
print("=" * 78)
print(f"Baseline (só mid, pool comum):  sharpe={sh_bl:+.2f}  n_sig={n_bl}  totPnL={100*pnls_bl.sum():+.2f}%")
print()
print("Standalone por horizonte:")
for name, m in standalone_metrics.items():
    print(
        f"  {name:>5s}  sharpe={m['sharpe']:+.2f}  n_sig={m['n_sig']:>4d}  "
        f"avg={100*m['avg_pnl']:+.3f}%  tot={100*m['tot_pnl']:+.2f}%"
    )
print()
print("Ensembles:")
best_rule, best_delta = None, -1e9
for name, r in ensemble_results.items():
    if name == "baseline_mid":
        continue
    delta = r["sharpe"] - sh_bl
    print(f"  {name:>17s}  sharpe={r['sharpe']:+.2f}  Δ_vs_baseline={delta:+.2f}  n_sig={r['n_sig']}")
    if delta > best_delta:
        best_delta = delta
        best_rule = name

print()
print(f"Melhor regra: {best_rule}  (Δ Sharpe vs baseline = {best_delta:+.2f})")
if best_delta > 0.10:
    veredito = "INTEGRAR — ganho material de Sharpe."
elif best_delta > 0:
    veredito = "MARGINAL — ganho pequeno; testar com mais folds antes de integrar."
else:
    veredito = "DESCARTAR — ensemble não melhora baseline."
print(f"Recomendação: {veredito}")
