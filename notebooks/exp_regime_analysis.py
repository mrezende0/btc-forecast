# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # EXP — Performance por regime de mercado (BULL / BEAR / CHOP)
#
# Roda walk-forward DUAL-HORIZON AND (mid=12, long=18, thr=0.35) entre
# 2023Q1 e 2026Q2 e decompõe a performance por regime mensal:
#
#   - BULL: BTC subiu > 5% no mês
#   - BEAR: BTC caiu  > 5% no mês
#   - CHOP: variação entre -5% e +5%
#
# Cada trade é classificado pelo regime do MÊS de abertura.
# Reporta: trades, win rate, avg PnL, PnL total, Sharpe (anualizado), MaxDD.

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
from pipeline import model as mdl

TIMEFRAME = mdl.TIMEFRAME_MIN          # 240 (4h)
H_MID = mdl.HORIZON_BARS               # 12 bars
H_LONG = mdl.HORIZON_BARS_LONG         # 18 bars
ATR_MULT = mdl.ATR_MULT                # 3.0
THRESHOLD = mdl.SIGNAL_THRESHOLD       # 0.35
COST = 0.0008                          # 0.08%
INITIAL_CAPITAL = 1000.0
BARS_PER_YEAR = 6 * 365                # 4h => 6 bars/day

PARAMS = mdl.LGB_PARAMS
N_ROUNDS = mdl.N_ROUNDS

# %% [markdown]
# ## 1. Build matrix + label sets (mid e long)

# %%
print("Building matrix @ 4h...")
df_base = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME, lag=1).drop_nulls(subset=["atr_14"])
print(f"shape: {df_base.shape}")
print(
    "range: "
    f"{datetime.fromtimestamp(df_base['open_time'].min()/1000, tz=timezone.utc).date()} → "
    f"{datetime.fromtimestamp(df_base['open_time'].max()/1000, tz=timezone.utc).date()}"
)


def build_mat(h: int) -> tuple[pd.DataFrame, list[str]]:
    lbl = lab.triple_barrier(df_base, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=h)
    lbl = lbl.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y_bin"))
    fcols = [
        c for c in lbl.columns
        if c not in feat.LAG_SAFE_EXCLUDE
        and c not in {"label", "hit_bar", "barrier_ret", "upper_px", "lower_px", "y_bin"}
    ]
    m = lbl.select(["open_time", "close", "y_bin", "barrier_ret", *fcols]).drop_nulls(
        subset=fcols + ["y_bin"]
    ).to_pandas()
    m["dt"] = m["open_time"].apply(lambda ms: datetime.fromtimestamp(ms / 1000, tz=timezone.utc))
    m["quarter"] = m["dt"].dt.to_period("Q")
    m["month"] = m["dt"].dt.to_period("M")
    return m, fcols


mat_mid, fc_mid = build_mat(H_MID)
mat_long, fc_long = build_mat(H_LONG)
assert fc_mid == fc_long, "feature cols devem coincidir"
feature_cols = fc_mid
print(f"Features: {len(feature_cols)}  |  mid rows: {len(mat_mid)}  |  long rows: {len(mat_long)}")

# %% [markdown]
# ## 2. Define regimes mensais de BTC
#
# Usa close mensal (final do mês) e variação mês a mês. Aplica isso a cada
# bar pelo mês a que pertence.

# %%
# Pega close diário e calcula close mensal
daily_close = (
    df_base.select(["open_time", "close"])
    .with_columns(
        pl.from_epoch(pl.col("open_time"), time_unit="ms").alias("dt")
    )
    .to_pandas()
)
daily_close["dt"] = pd.to_datetime(daily_close["dt"], utc=True)
daily_close["month"] = daily_close["dt"].dt.to_period("M")
# Última cotação do mês como "close do mês"
monthly = daily_close.groupby("month")["close"].last().reset_index()
monthly["close_prev"] = monthly["close"].shift(1)
monthly["ret_month"] = monthly["close"] / monthly["close_prev"] - 1.0

def classify(r):
    if pd.isna(r):
        return "UNK"
    if r > 0.05:
        return "BULL"
    if r < -0.05:
        return "BEAR"
    return "CHOP"

monthly["regime"] = monthly["ret_month"].apply(classify)
print("\nDistribuição de meses por regime (toda a história):")
print(monthly["regime"].value_counts())
print("\nÚltimos 12 meses:")
print(monthly.tail(12).to_string(index=False))

regime_lut = dict(zip(monthly["month"], monthly["regime"]))

# %% [markdown]
# ## 3. Walk-forward DUAL-HORIZON AND, quarterly

# %%
quarters = [q for q in sorted(mat_mid["quarter"].unique()) if q.start_time.year >= 2023]
print(f"\nQuarters de teste: {len(quarters)} ({quarters[0]} → {quarters[-1]})")

proba_mid_by_ot: dict[int, float] = {}
proba_long_by_ot: dict[int, float] = {}


def run_walkforward(mat: pd.DataFrame, h: int, store: dict[int, float]):
    for q in quarters:
        test_mask = mat["quarter"] == q
        test_idx = mat.index[test_mask].tolist()
        if not test_idx:
            continue
        test_start = test_idx[0]
        train_end = test_start - h
        test_use_start = test_start + h
        if train_end < 500 or test_use_start >= test_idx[-1]:
            continue
        train_idx = list(range(0, train_end))
        test_use_idx = [i for i in test_idx if i >= test_use_start]

        X_tr = mat.iloc[train_idx][feature_cols].values
        y_tr = mat.iloc[train_idx]["y_bin"].values
        X_te = mat.iloc[test_use_idx][feature_cols].values
        ot_te = mat.iloc[test_use_idx]["open_time"].values

        t0 = time.time()
        dtr = lgb.Dataset(X_tr, y_tr)
        model = lgb.train(PARAMS, dtr, num_boost_round=N_ROUNDS)
        proba = model.predict(X_te)
        for ot, p in zip(ot_te, proba):
            store[int(ot)] = float(p)
        print(f"  h={h:>2d}  {str(q):>8s}  tr={len(train_idx):>5d}  te={len(test_use_idx):>4d}  ({time.time()-t0:.1f}s)")


print("\n--- Walk-forward MID (h=12) ---")
run_walkforward(mat_mid, H_MID, proba_mid_by_ot)
print("\n--- Walk-forward LONG (h=18) ---")
run_walkforward(mat_long, H_LONG, proba_long_by_ot)

# %% [markdown]
# ## 4. Combina sinais (AND) e monta lista de trades

# %%
ots_common = sorted(set(proba_mid_by_ot) & set(proba_long_by_ot))
print(f"\nOTs comuns aos dois modelos: {len(ots_common)}")

ret_lut_mid = dict(zip(mat_mid["open_time"].astype(int), mat_mid["barrier_ret"]))
month_lut = dict(zip(mat_mid["open_time"].astype(int), mat_mid["month"]))
dt_lut = dict(zip(mat_mid["open_time"].astype(int), mat_mid["dt"]))

rows = []
for ot in ots_common:
    p_m = proba_mid_by_ot[ot]
    p_l = proba_long_by_ot[ot]
    sig = (p_m > THRESHOLD) and (p_l > THRESHOLD)
    if not sig:
        continue
    ret = ret_lut_mid.get(ot)
    if ret is None or pd.isna(ret):
        continue
    month = month_lut.get(ot)
    regime = regime_lut.get(month, "UNK")
    rows.append({
        "open_time": ot,
        "dt": dt_lut.get(ot),
        "month": str(month),
        "regime": regime,
        "proba_mid": p_m,
        "proba_long": p_l,
        "ret": ret,
        "pnl_pct": ret - COST,
        "win": (ret - COST) > 0,
    })

trades = pd.DataFrame(rows).sort_values("open_time").reset_index(drop=True)
print(f"Total de trades dual-horizon AND: {len(trades)}")

# Equity curve full-size $1000
equity = INITIAL_CAPITAL
eq_curve = []
for _, t in trades.iterrows():
    equity *= (1 + t["pnl_pct"])
    eq_curve.append(equity)
trades["equity"] = eq_curve
print(f"Equity final: ${equity:,.2f}  (return total: {100*(equity/INITIAL_CAPITAL-1):+.1f}%)")

# %% [markdown]
# ## 5. % do tempo em cada regime no período testado

# %%
# Restringe meses ao intervalo dos trades (período walk-forward de teste)
months_tested = sorted(set(month_lut[ot] for ot in ots_common))
months_tested_df = monthly[monthly["month"].isin(months_tested)].copy()
print("\n% do tempo (meses) em cada regime no período de teste:")
regime_time_share = months_tested_df["regime"].value_counts(normalize=True).sort_index()
for r, pct in regime_time_share.items():
    n = int(months_tested_df["regime"].value_counts()[r])
    print(f"  {r:>5s}  {n:>3d} meses  ({100*pct:>4.1f}%)")

# %% [markdown]
# ## 6. Métricas por regime

# %%
def sharpe_ann(pnls: np.ndarray, n_years: float) -> float:
    if len(pnls) < 2 or pnls.std(ddof=1) == 0 or n_years <= 0:
        return 0.0
    trades_per_year = len(pnls) / n_years
    return float(pnls.mean() / pnls.std(ddof=1) * np.sqrt(trades_per_year))


def max_dd(equity_arr: np.ndarray) -> float:
    if len(equity_arr) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_arr)
    dd = (equity_arr - peak) / peak
    return float(dd.min())


# Período total de teste em anos
ts_min = trades["open_time"].min()
ts_max = trades["open_time"].max()
years_total = (ts_max - ts_min) / 1000 / 86400 / 365 if len(trades) > 1 else 1.0

# % PnL por regime (em PnL aditivo simples para "% do PnL total")
trades["pnl_pct_simple"] = trades["pnl_pct"]
sum_pnl_total = trades["pnl_pct_simple"].sum()

print("\n" + "=" * 92)
print(f"PERFORMANCE POR REGIME — DUAL-HORIZON AND (thr={THRESHOLD}, cost={COST*100:.2f}%, capital=${INITIAL_CAPITAL})")
print("=" * 92)
header = f"{'Regime':>6s}  {'Trades':>7s}  {'Win%':>6s}  {'AvgPnL':>8s}  {'TotPnL':>9s}  {'Sharpe':>7s}  {'MaxDD':>7s}  {'%PnL':>7s}  {'%Tempo':>7s}"
print(header)
print("-" * len(header))

summary_rows = []
for regime in ["BULL", "CHOP", "BEAR"]:
    grp = trades[trades["regime"] == regime].copy()
    n = len(grp)
    if n == 0:
        print(f"  {regime:>4s}  {0:>7d}  {'-':>6s}  {'-':>8s}  {'-':>9s}  {'-':>7s}  {'-':>7s}  {'-':>7s}  ")
        summary_rows.append({"regime": regime, "n": 0})
        continue
    pnls = grp["pnl_pct"].values
    win = (pnls > 0).mean()
    avg = pnls.mean()
    tot = pnls.sum()
    # Anos restritos ao intervalo dos trades do regime
    yrs_reg = max(1e-9, (grp["open_time"].max() - grp["open_time"].min()) / 1000 / 86400 / 365)
    sh = sharpe_ann(pnls, yrs_reg)
    # Intra-regime equity (composto só nos trades do regime)
    eq_reg = INITIAL_CAPITAL * np.cumprod(1 + pnls)
    dd_reg = max_dd(eq_reg)
    pct_pnl = tot / sum_pnl_total if sum_pnl_total else float("nan")
    pct_time = regime_time_share.get(regime, 0.0)
    print(
        f"  {regime:>4s}  {n:>7d}  {100*win:>5.1f}%  {100*avg:>+7.3f}%  {100*tot:>+8.2f}%  "
        f"{sh:>+6.2f}  {100*dd_reg:>+6.1f}%  {100*pct_pnl:>+6.1f}%  {100*pct_time:>6.1f}%"
    )
    summary_rows.append({
        "regime": regime, "n": n, "win": win, "avg": avg, "tot": tot,
        "sharpe": sh, "maxdd": dd_reg, "pct_pnl": pct_pnl, "pct_time": pct_time,
    })

# Total agregado
pnls_all = trades["pnl_pct"].values
sh_all = sharpe_ann(pnls_all, years_total)
dd_all = max_dd(trades["equity"].values)
print("-" * len(header))
print(
    f"  {'TOT':>4s}  {len(trades):>7d}  {100*(pnls_all>0).mean():>5.1f}%  "
    f"{100*pnls_all.mean():>+7.3f}%  {100*pnls_all.sum():>+8.2f}%  "
    f"{sh_all:>+6.2f}  {100*dd_all:>+6.1f}%  {100*1.0:>+6.1f}%  {100:>6.1f}%"
)

# %% [markdown]
# ## 7. Pior fold mensal (período mais difícil)

# %%
trades["month_str"] = trades["dt"].dt.to_period("M").astype(str)
monthly_pnl = trades.groupby("month_str").agg(
    n=("pnl_pct", "size"),
    tot=("pnl_pct", "sum"),
    win=("win", "mean"),
).reset_index()
monthly_pnl = monthly_pnl.merge(
    monthly.assign(month_str=monthly["month"].astype(str))[["month_str", "regime", "ret_month"]],
    on="month_str", how="left"
)
worst = monthly_pnl.sort_values("tot").head(5)
best = monthly_pnl.sort_values("tot", ascending=False).head(5)
print("\nPiores 5 meses (PnL):")
for _, r in worst.iterrows():
    print(f"  {r['month_str']}  regime={r['regime']:>4s}  trades={int(r['n']):>2d}  win={100*r['win']:>4.1f}%  tot={100*r['tot']:+6.2f}%  (BTC mês: {100*r['ret_month']:+.1f}%)")
print("\nMelhores 5 meses (PnL):")
for _, r in best.iterrows():
    print(f"  {r['month_str']}  regime={r['regime']:>4s}  trades={int(r['n']):>2d}  win={100*r['win']:>4.1f}%  tot={100*r['tot']:+6.2f}%  (BTC mês: {100*r['ret_month']:+.1f}%)")

# Pior janela trimestral
trades["quarter_str"] = trades["dt"].dt.to_period("Q").astype(str)
quarterly_pnl = trades.groupby("quarter_str").agg(
    n=("pnl_pct", "size"),
    tot=("pnl_pct", "sum"),
    win=("win", "mean"),
).reset_index().sort_values("tot")
print("\nPior trimestre:")
print(quarterly_pnl.head(3).to_string(index=False))
print("\nMelhor trimestre:")
print(quarterly_pnl.tail(3).to_string(index=False))

# %% [markdown]
# ## 8. Simulação de filtro de regime (apenas BULL+CHOP)

# %%
print("\n" + "=" * 92)
print("SIMULAÇÃO — desligar trades em BEAR (rodar só em BULL+CHOP)")
print("=" * 92)

for filter_name, mask in [
    ("baseline (all)", trades["regime"].notna()),
    ("só BULL", trades["regime"] == "BULL"),
    ("só CHOP", trades["regime"] == "CHOP"),
    ("BULL + CHOP", trades["regime"].isin(["BULL", "CHOP"])),
    ("sem BEAR", trades["regime"] != "BEAR"),
]:
    sub = trades[mask].sort_values("open_time")
    if len(sub) == 0:
        print(f"  {filter_name:>15s}  (sem trades)")
        continue
    pnls = sub["pnl_pct"].values
    eq = INITIAL_CAPITAL * np.cumprod(1 + pnls)
    yrs = max(1e-9, (sub["open_time"].max() - sub["open_time"].min()) / 1000 / 86400 / 365)
    sh = sharpe_ann(pnls, yrs)
    dd = max_dd(eq)
    print(
        f"  {filter_name:>15s}  n={len(sub):>4d}  win={100*(pnls>0).mean():>4.1f}%  "
        f"avg={100*pnls.mean():>+6.3f}%  tot={100*pnls.sum():>+7.2f}%  "
        f"equity=${eq[-1]:>8,.2f}  sharpe={sh:>+5.2f}  MDD={100*dd:>+6.1f}%"
    )

# %% [markdown]
# ## 9. Resumo final

# %%
print("\n" + "=" * 92)
print("RESUMO FINAL")
print("=" * 92)
print(f"Trades totais: {len(trades)}  |  Período: {years_total:.2f} anos  |  Capital final: ${equity:,.2f}")
print(f"Sharpe global: {sh_all:+.2f}  |  MaxDD: {100*dd_all:+.1f}%")
print()
print("Linha por regime salva em summary_rows; tabela acima.")
