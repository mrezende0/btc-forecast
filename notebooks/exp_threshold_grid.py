# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # EXP — Threshold sensitivity (DUAL-HORIZON)
#
# Modelo em produção: AND(mid > thr, long > thr) com thr=0.35.
# Este experimento varre o grid 5x5:
#     mid_thr  in {0.30, 0.35, 0.40, 0.45, 0.50}
#     long_thr in {0.30, 0.35, 0.40, 0.45, 0.50}
#
# Pipeline:
#   1. Treina UMA vez por trimestre (mid h=12 e long h=18) — walk-forward expanding
#   2. Armazena probas alinhadas por open_time
#   3. Avalia 25 combos sem retreinar (só re-aplicando thresholds nas probas)
#
# Metricas por combo:
#   - n_sig, win_rate, avg_pnl, tot_pnl
#   - Sharpe (anualizado por trades/ano)
#   - capital final ($1000 inicial, full position por trade)
#   - MaxDD (no curve de equity por trade)
#
# PnL de cada trade = ret_mid (horizonte do trade real, 48h) - COST (0.08% round-trip).

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
COST = 0.0015            # 0.15% round-trip (Binance taker 0.10% × 2 + slippage real)
INITIAL_CAPITAL = 1000.0

HORIZON_MID = 12   # 48h
HORIZON_LONG = 18  # 72h

GRID = [0.30, 0.35, 0.40, 0.45, 0.50]

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
# ## 1. Build feature matrix (4h, uma vez)

# %%
print("Building feature matrix @ 4h...")
df_base = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME, lag=1)
df_base = df_base.drop_nulls(subset=["atr_14"])
print(f"shape: {df_base.shape}")
print(
    f"range: "
    f"{datetime.fromtimestamp(df_base['open_time'].min()/1000, tz=timezone.utc).date()} -> "
    f"{datetime.fromtimestamp(df_base['open_time'].max()/1000, tz=timezone.utc).date()}"
)

# %% [markdown]
# ## 2. Gera dois sets de labels (mid h=12, long h=18)

# %%
labels_mid = lab.triple_barrier(df_base, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=HORIZON_MID)
labels_long = lab.triple_barrier(df_base, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=HORIZON_LONG)
labels_mid = labels_mid.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y_bin"))
labels_long = labels_long.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y_bin"))

print(f"\nlabels_mid:  {labels_mid.height} rows  | base rate LONG_WIN = {100*labels_mid['y_bin'].mean():.1f}%")
print(f"labels_long: {labels_long.height} rows  | base rate LONG_WIN = {100*labels_long['y_bin'].mean():.1f}%")

# %% [markdown]
# ## 3. Feature set comum

# %%
sample = labels_mid
feature_cols = [
    c for c in sample.columns
    if c not in feat.LAG_SAFE_EXCLUDE
    and c not in {"label", "hit_bar", "barrier_ret", "upper_px", "lower_px", "y_bin"}
]
print(f"Features: {len(feature_cols)}")

# %% [markdown]
# ## 4. Constroi matrizes pandas alinhaveis (uma por horizonte)

# %%
def to_pandas_mat(lbl_pl: pl.DataFrame) -> pd.DataFrame:
    m = lbl_pl.select(["open_time", "close", "y_bin", "barrier_ret", *feature_cols]).drop_nulls(
        subset=feature_cols + ["y_bin"]
    ).to_pandas()
    m["dt"] = m["open_time"].apply(lambda ms: datetime.fromtimestamp(ms / 1000, tz=timezone.utc))
    m["quarter"] = m["dt"].dt.to_period("Q")
    return m

mat_mid = to_pandas_mat(labels_mid)
mat_long = to_pandas_mat(labels_long)

print(f"mat_mid:  {len(mat_mid)} rows | {mat_mid['dt'].min().date()} -> {mat_mid['dt'].max().date()}")
print(f"mat_long: {len(mat_long)} rows | {mat_long['dt'].min().date()} -> {mat_long['dt'].max().date()}")

# %% [markdown]
# ## 5. Walk-forward expanding quarterly — treina mid + long uma vez por trimestre
#
# Armazena proba_mid[open_time] e proba_long[open_time] para reuso no grid.

# %%
quarters = [q for q in sorted(mat_mid["quarter"].unique()) if q.start_time.year >= 2023]
print(f"\nQuarters: {len(quarters)} ({quarters[0]} -> {quarters[-1]})")

proba_mid_by_ot: dict[int, float] = {}
proba_long_by_ot: dict[int, float] = {}

t_total = time.time()
for q in quarters:
    for name, m, h, store in [
        ("mid", mat_mid, HORIZON_MID, proba_mid_by_ot),
        ("long", mat_long, HORIZON_LONG, proba_long_by_ot),
    ]:
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

        t0 = time.time()
        dtr = lgb.Dataset(X_tr, y_tr)
        model = lgb.train(PARAMS, dtr, num_boost_round=N_ROUNDS)
        proba = model.predict(X_te)
        for ot, p in zip(ot_te, proba):
            store[int(ot)] = float(p)
        dt_train = time.time() - t0
        # print compacto
        # (deixa silencioso para nao poluir)
    print(f"  {str(q)} ok | mid_probas={len(proba_mid_by_ot):>5d}  long_probas={len(proba_long_by_ot):>5d}")

print(f"\nTreino total: {time.time()-t_total:.1f}s")

# %% [markdown]
# ## 6. Alinha probas e retornos por open_time
#
# Pool = open_times presentes em AMBOS os horizontes (mid e long).
# PnL = ret_mid (horizonte de trade real, 48h).

# %%
ots_common = sorted(set(proba_mid_by_ot.keys()) & set(proba_long_by_ot.keys()))
print(f"Open_times no pool comum: {len(ots_common)}")

p_mid = np.array([proba_mid_by_ot[o] for o in ots_common])
p_long = np.array([proba_long_by_ot[o] for o in ots_common])

ret_lut = dict(zip(mat_mid["open_time"].astype(int), mat_mid["barrier_ret"]))
ret_ref = np.array([ret_lut[int(o)] for o in ots_common], dtype=float)

# Janela em anos para anualizar Sharpe
n_years_pool = max(1e-9, (ots_common[-1] - ots_common[0]) / 1000 / 86400 / 365)
print(f"Pool: {n_years_pool:.2f} anos")

# %% [markdown]
# ## 7. Funcoes de metrica

# %%
def sharpe_trades(pnls: np.ndarray, trades_per_year: float) -> float:
    if len(pnls) < 2 or pnls.std(ddof=1) == 0:
        return 0.0
    return float(pnls.mean() / pnls.std(ddof=1) * np.sqrt(trades_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    """MaxDD em % do peak (negativo)."""
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())


def compound_capital(pnls_net: np.ndarray, initial: float = INITIAL_CAPITAL) -> tuple[float, np.ndarray]:
    """Full position por trade -> capital_{t+1} = capital_t * (1 + pnl_net_t)."""
    eq = initial * np.cumprod(1.0 + pnls_net)
    final = float(eq[-1]) if len(eq) else initial
    return final, eq


def eval_combo(mid_thr: float, long_thr: float) -> dict:
    mask = (p_mid > mid_thr) & (p_long > long_thr)
    n_sig = int(mask.sum())
    if n_sig == 0:
        return dict(
            mid_thr=mid_thr, long_thr=long_thr, n_sig=0, win_rate=0.0,
            avg_pnl=0.0, tot_pnl=0.0, sharpe=0.0,
            capital_final=INITIAL_CAPITAL, return_pct=0.0, max_dd=0.0,
        )
    pnls_net = ret_ref[mask] - COST
    trades_per_year = n_sig / n_years_pool
    sh = sharpe_trades(pnls_net, trades_per_year)
    cap_final, equity = compound_capital(pnls_net)
    return dict(
        mid_thr=mid_thr,
        long_thr=long_thr,
        n_sig=n_sig,
        win_rate=float((pnls_net > 0).mean()),
        avg_pnl=float(pnls_net.mean()),
        tot_pnl=float(pnls_net.sum()),  # PnL simples agregado
        sharpe=sh,
        capital_final=cap_final,
        return_pct=(cap_final / INITIAL_CAPITAL - 1.0) * 100.0,
        max_dd=max_drawdown(equity) * 100.0,  # em %
    )

# %% [markdown]
# ## 8. Roda o grid 5x5

# %%
rows = []
for mt in GRID:
    for lt in GRID:
        rows.append(eval_combo(mt, lt))

res = pd.DataFrame(rows)

# %% [markdown]
# ## 9. Matrizes 5x5 — Sharpe / Capital final / MaxDD / n_sig

# %%
def pivot_metric(col: str) -> pd.DataFrame:
    return res.pivot(index="mid_thr", columns="long_thr", values=col)

def fmt_grid(pv: pd.DataFrame, fmt: str, title: str) -> str:
    lines = [title, "=" * len(title)]
    header = "mid\\long".rjust(10) + "  " + "  ".join(f"{c:>8.2f}" for c in pv.columns)
    lines.append(header)
    lines.append("-" * len(header))
    for idx, row in pv.iterrows():
        lines.append(f"{idx:>10.2f}  " + "  ".join(format(v, fmt).rjust(8) for v in row.values))
    return "\n".join(lines)

print("\n")
print(fmt_grid(pivot_metric("sharpe"), "+.2f", "SHARPE (anualizado)"))
print("\n")
print(fmt_grid(pivot_metric("capital_final"), ".1f", "CAPITAL FINAL ($1000 inicial)"))
print("\n")
print(fmt_grid(pivot_metric("return_pct"), "+.1f", "RETORNO TOTAL (%)"))
print("\n")
print(fmt_grid(pivot_metric("max_dd"), "+.1f", "MAX DRAWDOWN (%)"))
print("\n")
print(fmt_grid(pivot_metric("n_sig"), ".0f", "N_SIGNALS"))
print("\n")
print(fmt_grid(pivot_metric("win_rate") * 100, ".1f", "WIN RATE (%)"))

# %% [markdown]
# ## 10. Heatmap ASCII do Sharpe (rapido de bater o olho)

# %%
def heatmap_ascii(pv: pd.DataFrame, vmin: float = None, vmax: float = None) -> str:
    chars = " .:-=+*#%@"
    vals = pv.values.astype(float)
    vmin = vmin if vmin is not None else np.nanmin(vals)
    vmax = vmax if vmax is not None else np.nanmax(vals)
    rng = max(1e-9, vmax - vmin)
    lines = ["HEATMAP — Sharpe (escuro=baixo / claro=alto)", f"min={vmin:+.2f}  max={vmax:+.2f}"]
    header = "mid\\long".rjust(10) + "  " + "  ".join(f"{c:>5.2f}" for c in pv.columns)
    lines.append(header)
    for idx, row in pv.iterrows():
        cells = []
        for v in row.values:
            t = (v - vmin) / rng
            ch = chars[min(len(chars)-1, max(0, int(t * (len(chars)-1))))]
            cells.append(f"  {ch}{ch}{ch}  ")
        lines.append(f"{idx:>10.2f} " + "".join(cells))
    return "\n".join(lines)

print(heatmap_ascii(pivot_metric("sharpe")))

# %% [markdown]
# ## 11. Top 3 por Sharpe + quem perde menos em MaxDD

# %%
print("\n" + "=" * 78)
print("TOP 3 por SHARPE")
print("=" * 78)
top_sh = res.sort_values("sharpe", ascending=False).head(3)
for _, r in top_sh.iterrows():
    print(
        f"  mid={r['mid_thr']:.2f} long={r['long_thr']:.2f}  "
        f"Sharpe={r['sharpe']:+.2f}  cap=${r['capital_final']:>7.2f}  "
        f"ret={r['return_pct']:+6.1f}%  MaxDD={r['max_dd']:+6.1f}%  "
        f"n_sig={int(r['n_sig'])}  win={100*r['win_rate']:.1f}%"
    )

print("\n" + "=" * 78)
print("MENOR MAX DRAWDOWN (mais proximo de zero, com n_sig >= 10)")
print("=" * 78)
filt = res[res["n_sig"] >= 10].copy()
top_dd = filt.sort_values("max_dd", ascending=False).head(3)  # max_dd e negativo; maior = melhor
for _, r in top_dd.iterrows():
    print(
        f"  mid={r['mid_thr']:.2f} long={r['long_thr']:.2f}  "
        f"MaxDD={r['max_dd']:+6.1f}%  Sharpe={r['sharpe']:+.2f}  "
        f"cap=${r['capital_final']:>7.2f}  ret={r['return_pct']:+6.1f}%  "
        f"n_sig={int(r['n_sig'])}"
    )

# %% [markdown]
# ## 12. Comparacao vs baseline atual (0.35 / 0.35)

# %%
baseline = res[(res["mid_thr"] == 0.35) & (res["long_thr"] == 0.35)].iloc[0]
best = res.sort_values("sharpe", ascending=False).iloc[0]

print("\n" + "=" * 78)
print("BASELINE (0.35 / 0.35) vs MELHOR (por Sharpe)")
print("=" * 78)
print(
    f"  Baseline: mid=0.35 long=0.35  "
    f"Sharpe={baseline['sharpe']:+.2f}  cap=${baseline['capital_final']:>7.2f}  "
    f"ret={baseline['return_pct']:+6.1f}%  MaxDD={baseline['max_dd']:+6.1f}%  "
    f"n_sig={int(baseline['n_sig'])}"
)
print(
    f"  Melhor:   mid={best['mid_thr']:.2f} long={best['long_thr']:.2f}  "
    f"Sharpe={best['sharpe']:+.2f}  cap=${best['capital_final']:>7.2f}  "
    f"ret={best['return_pct']:+6.1f}%  MaxDD={best['max_dd']:+6.1f}%  "
    f"n_sig={int(best['n_sig'])}"
)

delta_sh = best["sharpe"] - baseline["sharpe"]
delta_cap = best["capital_final"] - baseline["capital_final"]
delta_dd = best["max_dd"] - baseline["max_dd"]
print(
    f"\n  Delta vs baseline: dSharpe={delta_sh:+.2f}  "
    f"dCapital=${delta_cap:+.2f}  dMaxDD={delta_dd:+.1f}pp"
)

if delta_sh > 0.15 and best["n_sig"] >= 20:
    veredito = "TROCAR — ganho material de Sharpe com amostra suficiente."
elif delta_sh > 0.05:
    veredito = "MARGINAL — ganho pequeno; manter 0.35 ate ver mais OOS."
else:
    veredito = "MANTER 0.35 — sem ganho real."
print(f"\nRECOMENDACAO: {veredito}")

# %% [markdown]
# ## 13. Dump CSV pra inspecao posterior

# %%
out = Path("notebooks") / "exp_threshold_grid_results.csv"
res.to_csv(out, index=False)
print(f"\nResultados salvos em: {out}")
