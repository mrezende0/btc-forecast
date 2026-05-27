# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # EXP — Position Sizing Sensitivity
#
# Testa 5 esquemas de tamanho de posicao sobre o sinal dual-horizon AND
# (mid=12, long=18, threshold=0.35) em walk-forward purgado.
#
# Esquemas:
#   A) FULL      — 100% do capital por trade (producao atual)
#   B) HALF      — 50% do capital por trade
#   C) QUARTER   — 25% do capital por trade
#   D) RISK-1PCT — sizing dinamico para stop = -1% do capital
#                  tamanho_btc = (0.01 * capital) / (entry - stop)
#                  cap em 100% do capital (sem alavancagem)
#   E) KELLY_FRAC — 2.5% do capital por trade (~1/4 Kelly @ p=0.55, b=1)
#
# Regras:
#   - Sinal: dual-horizon AND (proba_mid > 0.35 AND proba_long > 0.35)
#   - 1 posicao por vez (sinais durante posicao aberta = ignorados)
#   - Capital atualiza apos cada trade fechado
#   - Retreina a cada 90 dias com purged walk-forward
#   - Periodo: 2023-01-01 -> presente

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

# %% [markdown]
# ## Constantes (alinhadas com pipeline.model)

# %%
TIMEFRAME = 240          # 4h
ATR_MULT = 3.0
COST = 0.0008            # 0.08% round-trip
THRESHOLD = 0.35
HORIZON_MID = 12         # 48h
HORIZON_LONG = 18        # 72h
CAPITAL_INIT = 1_000.0
BARS_PER_DAY = 6         # 4h => 6 bars/day
RETRAIN_EVERY_DAYS = 90
RETRAIN_EVERY_BARS = RETRAIN_EVERY_DAYS * BARS_PER_DAY
TRADES_PER_YEAR_REF = 365 / 7  # estimativa grossa p/ Sharpe (ajustada no fim)

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

print(f"Capital inicial: ${CAPITAL_INIT:.2f}")
print(f"Retreino a cada {RETRAIN_EVERY_DAYS} dias ({RETRAIN_EVERY_BARS} bars de 4h)")
print(f"Threshold dual-horizon AND: {THRESHOLD}")

# %% [markdown]
# ## 1. Build feature matrix + labels para os 2 horizontes

# %%
print("\nBuilding feature matrix @ 4h...")
df_base = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME, lag=1)
df_base = df_base.drop_nulls(subset=["atr_14"])
print(f"shape: {df_base.shape}")
print(
    f"range: "
    f"{datetime.fromtimestamp(df_base['open_time'].min()/1000, tz=timezone.utc).date()} -> "
    f"{datetime.fromtimestamp(df_base['open_time'].max()/1000, tz=timezone.utc).date()}"
)

# Labels para mid e long
print("\nLabeling mid (h=12)...")
lbl_mid = lab.triple_barrier(df_base, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=HORIZON_MID)
lbl_mid = lbl_mid.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y_bin"))

print("Labeling long (h=18)...")
lbl_long = lab.triple_barrier(df_base, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=HORIZON_LONG)
lbl_long = lbl_long.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y_bin"))

feature_cols = [
    c for c in lbl_mid.columns
    if c not in feat.LAG_SAFE_EXCLUDE
    and c not in {"label", "hit_bar", "barrier_ret", "upper_px", "lower_px", "y_bin"}
]
print(f"Features: {len(feature_cols)}")

# Cria matrizes pandas alinhadas por open_time
base_cols_mid = ["open_time", "close", "y_bin", "barrier_ret", "hit_bar", "upper_px", "lower_px"]
sel_mid = base_cols_mid + [c for c in feature_cols if c not in base_cols_mid]
m_mid = lbl_mid.select(sel_mid).drop_nulls(subset=feature_cols + ["y_bin"]).to_pandas().reset_index(drop=True)
base_cols_long = ["open_time", "close", "y_bin", "barrier_ret", "hit_bar"]
sel_long = base_cols_long + [c for c in feature_cols if c not in base_cols_long]
m_long = lbl_long.select(sel_long).drop_nulls(subset=feature_cols + ["y_bin"]).to_pandas().reset_index(drop=True)

m_mid["dt"] = pd.to_datetime(m_mid["open_time"], unit="ms", utc=True)
m_long["dt"] = pd.to_datetime(m_long["open_time"], unit="ms", utc=True)

print(f"m_mid:  {len(m_mid)} rows, range {m_mid['dt'].min().date()} -> {m_mid['dt'].max().date()}")
print(f"m_long: {len(m_long)} rows, range {m_long['dt'].min().date()} -> {m_long['dt'].max().date()}")

# %% [markdown]
# ## 2. Walk-forward purgado: retreina cada 90 dias, gera probas dual-horizon
#
# Para cada janela de teste de 90 dias:
#   - Treina mid e long em [0, train_end_mid/long)
#   - Prediz em [test_use_start, test_end)
#   - Purge entre train e test = horizon_bars (12 ou 18)

# %%
START_DATE = pd.Timestamp("2023-01-01", tz="utc")
mid_start_idx = m_mid[m_mid["dt"] >= START_DATE].index.min()
print(f"\nStart test idx (mid): {mid_start_idx}  dt={m_mid.loc[mid_start_idx, 'dt'].date()}")

# Build lookup open_time -> idx para long
long_idx_by_ot = dict(zip(m_long["open_time"].astype(int), m_long.index))

proba_mid_arr = np.full(len(m_mid), np.nan)
proba_long_arr = np.full(len(m_mid), np.nan)

# Loop de retreino a cada RETRAIN_EVERY_BARS
n_retrains = 0
cursor = mid_start_idx
t_start = time.time()
while cursor < len(m_mid):
    window_end = min(cursor + RETRAIN_EVERY_BARS, len(m_mid))
    # ---- train mid
    train_end_mid = cursor - HORIZON_MID
    if train_end_mid < 500:
        cursor = window_end
        continue
    X_tr = m_mid.iloc[:train_end_mid][feature_cols].values
    y_tr = m_mid.iloc[:train_end_mid]["y_bin"].values
    model_mid = lgb.train(PARAMS, lgb.Dataset(X_tr, y_tr), num_boost_round=N_ROUNDS)

    # ---- train long (no mesmo cursor temporal — usa open_time)
    ot_cursor = int(m_mid.loc[cursor, "open_time"])
    # idx em m_long <= ot_cursor (purge horizon_long)
    long_idx_cursor = m_long[m_long["open_time"] >= ot_cursor].index.min()
    if pd.isna(long_idx_cursor):
        long_idx_cursor = len(m_long)
    train_end_long = int(long_idx_cursor) - HORIZON_LONG
    if train_end_long < 500:
        cursor = window_end
        continue
    X_tr_l = m_long.iloc[:train_end_long][feature_cols].values
    y_tr_l = m_long.iloc[:train_end_long]["y_bin"].values
    model_long = lgb.train(PARAMS, lgb.Dataset(X_tr_l, y_tr_l), num_boost_round=N_ROUNDS)

    # ---- predict no janelao [cursor, window_end)
    # Aplica purge de horizon_mid: o primeiro idx valido eh cursor (porque train terminou em cursor - h_mid)
    X_te = m_mid.iloc[cursor:window_end][feature_cols].values
    p_mid = model_mid.predict(X_te)
    proba_mid_arr[cursor:window_end] = p_mid

    # Para long: precisamos prever na mesma open_time de cada bar de mid
    ots_te = m_mid.iloc[cursor:window_end]["open_time"].astype(int).values
    for k, ot in enumerate(ots_te):
        li = long_idx_by_ot.get(int(ot))
        if li is None:
            continue
        # purge: so prediz se li >= train_end_long + HORIZON_LONG
        if li < train_end_long + HORIZON_LONG:
            continue
        xl = m_long.iloc[li:li+1][feature_cols].values
        proba_long_arr[cursor + k] = float(model_long.predict(xl)[0])

    n_retrains += 1
    print(
        f"  retrain #{n_retrains:>2d}  cursor={cursor:>5d}  "
        f"dt={m_mid.loc[cursor, 'dt'].date()}  "
        f"window=[{cursor},{window_end})  "
        f"train_mid={train_end_mid}  train_long={train_end_long}  "
        f"({time.time() - t_start:.1f}s)"
    )
    cursor = window_end

print(f"\nTotal retreinos: {n_retrains}  tempo total: {time.time() - t_start:.1f}s")

# %% [markdown]
# ## 3. Gera sinais dual-horizon AND

# %%
m_mid["proba_mid"] = proba_mid_arr
m_mid["proba_long"] = proba_long_arr
m_mid["signal"] = (
    (m_mid["proba_mid"] > THRESHOLD)
    & (m_mid["proba_long"] > THRESHOLD)
).astype(bool)

valid = m_mid[m_mid["proba_mid"].notna() & m_mid["proba_long"].notna()].copy()
n_signals = int(valid["signal"].sum())
print(f"\nBars validos: {len(valid)}  sinais brutos dual-AND: {n_signals}")

# %% [markdown]
# ## 4. Simulacao trade-by-trade (1 posicao por vez)
#
# Para cada bar com sinal, se nao ha posicao aberta:
#   - entry = close[t]
#   - stop  = lower_px[t]  (close[t] - 3*atr[t])
#   - target= upper_px[t]
#   - exit_bar = t + hit_bar (clamp para t + horizon_mid)
#   - barrier_ret ja foi calculado
#
# Capital evolui apos cada exit. Sinais durante posicao aberta sao ignorados.

# %%
def simulate(scheme: str, signals_df: pd.DataFrame, capital_init: float) -> dict:
    """Simula com o esquema de sizing dado. Retorna metricas + curva de equity."""
    capital = capital_init
    equity_curve = []   # (dt, capital)
    trades = []         # list of dicts
    pos_open_until_idx = -1  # idx ate o qual ha posicao aberta (exclusivo)

    sig_idxs = signals_df.index[signals_df["signal"]].tolist()
    # Mapeia idx -> row do m_mid (sig_idxs sao indices do m_mid)
    for idx in sig_idxs:
        if idx < pos_open_until_idx:
            continue  # ja em posicao
        row = m_mid.loc[idx]
        entry = float(row["close"])
        upper = float(row["upper_px"])
        lower = float(row["lower_px"])
        ret = float(row["barrier_ret"]) if not pd.isna(row["barrier_ret"]) else 0.0
        hit_bar = row["hit_bar"]
        if pd.isna(hit_bar):
            bars_held = HORIZON_MID
        else:
            bars_held = int(hit_bar)

        # ---- sizing
        stop_distance = entry - lower  # > 0
        stop_pct = stop_distance / entry  # ~3*atr/close

        if scheme == "FULL":
            position_value = capital  # 100%
        elif scheme == "HALF":
            position_value = capital * 0.5
        elif scheme == "QUARTER":
            position_value = capital * 0.25
        elif scheme == "RISK-1PCT":
            # tamanho_btc = (0.01 * capital) / stop_distance
            # position_value = tamanho_btc * entry = (0.01 * capital * entry) / stop_distance
            # = 0.01 * capital / stop_pct
            if stop_pct <= 0:
                continue
            position_value = (0.01 * capital) / stop_pct
            # cap em 100% do capital (sem alavancagem)
            position_value = min(position_value, capital)
        elif scheme == "KELLY_FRAC":
            position_value = capital * 0.025  # 1/4 Kelly @ p=0.55, b=1 -> 0.10/4=0.025
        else:
            raise ValueError(f"scheme desconhecido: {scheme}")

        # ---- PnL do trade
        # cost incide no notional do trade (round-trip)
        gross_pnl = position_value * ret
        cost_dollar = position_value * COST
        net_pnl = gross_pnl - cost_dollar

        capital_before = capital
        capital += net_pnl

        trades.append({
            "idx": idx,
            "dt": row["dt"],
            "entry": entry,
            "ret": ret,
            "stop_pct": stop_pct,
            "position_value": position_value,
            "position_pct": position_value / capital_before,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "capital_before": capital_before,
            "capital_after": capital,
            "bars_held": bars_held,
        })
        equity_curve.append((row["dt"], capital))
        pos_open_until_idx = idx + bars_held + 1  # libera para novo trade no bar seguinte ao exit

    if not trades:
        return {
            "scheme": scheme,
            "capital_final": capital_init,
            "ret_pct": 0.0,
            "max_dd_pct": 0.0,
            "n_trades": 0,
            "win_rate": 0.0,
            "sharpe": 0.0,
            "trades": [],
            "equity": [],
        }

    trades_df = pd.DataFrame(trades)
    eq = pd.DataFrame(equity_curve, columns=["dt", "capital"])
    # max drawdown a partir da curva de equity (apos cada trade)
    # incluindo capital inicial
    eq_full = pd.concat(
        [pd.DataFrame([{"dt": signals_df["dt"].min(), "capital": capital_init}]), eq],
        ignore_index=True,
    )
    eq_full["peak"] = eq_full["capital"].cummax()
    eq_full["dd"] = (eq_full["capital"] - eq_full["peak"]) / eq_full["peak"]
    max_dd = float(eq_full["dd"].min())

    # Win rate (por trade liquido)
    wins = int((trades_df["net_pnl"] > 0).sum())
    win_rate = wins / len(trades_df)

    # Sharpe: usa retornos % por trade (net_pnl / capital_before), anualiza por trades/ano
    trade_returns = trades_df["net_pnl"] / trades_df["capital_before"]
    n_years = (trades_df["dt"].max() - trades_df["dt"].min()).total_seconds() / (365.25 * 24 * 3600)
    n_years = max(n_years, 1e-9)
    trades_per_year = len(trades_df) / n_years
    if trade_returns.std(ddof=1) > 0 and len(trade_returns) > 1:
        sharpe = float(trade_returns.mean() / trade_returns.std(ddof=1) * np.sqrt(trades_per_year))
    else:
        sharpe = 0.0

    ret_pct = (capital / capital_init - 1) * 100

    return {
        "scheme": scheme,
        "capital_final": capital,
        "ret_pct": ret_pct,
        "max_dd_pct": max_dd * 100,
        "n_trades": len(trades_df),
        "win_rate": win_rate * 100,
        "sharpe": sharpe,
        "trades_per_year": trades_per_year,
        "trades": trades_df,
        "equity": eq_full,
    }


# %%
schemes = ["FULL", "HALF", "QUARTER", "RISK-1PCT", "KELLY_FRAC"]
results = {}
for s in schemes:
    r = simulate(s, valid, CAPITAL_INIT)
    results[s] = r
    print(
        f"  {s:>10s}  capital=${r['capital_final']:>10,.2f}  "
        f"ret={r['ret_pct']:+7.2f}%  maxDD={r['max_dd_pct']:+6.2f}%  "
        f"n={r['n_trades']:>3d}  win={r['win_rate']:>4.1f}%  sharpe={r['sharpe']:+.2f}"
    )

# %% [markdown]
# ## 5. Tabela comparativa final

# %%
print("\n" + "=" * 90)
print("TABELA COMPARATIVA — Position Sizing Sensitivity (capital inicial $1,000)")
print("=" * 90)
hdr = f"{'Scheme':>11s}  {'Final Cap':>11s}  {'Ret %':>8s}  {'Max DD %':>9s}  {'#Trades':>8s}  {'Win %':>6s}  {'Sharpe':>7s}"
print(hdr)
print("-" * 90)
for s in schemes:
    r = results[s]
    print(
        f"{s:>11s}  ${r['capital_final']:>10,.2f}  {r['ret_pct']:>+7.2f}  "
        f"{r['max_dd_pct']:>+8.2f}  {r['n_trades']:>8d}  {r['win_rate']:>5.1f}  {r['sharpe']:>+7.2f}"
    )
print("=" * 90)

# Tamanho medio de posicao (em % do capital antes do trade) para os schemes dinamicos
print("\nTamanho medio de posicao (% do capital antes do trade):")
for s in schemes:
    r = results[s]
    if r["n_trades"] == 0:
        print(f"  {s:>10s}: n/a")
        continue
    tdf = r["trades"]
    avg_pct = tdf["position_pct"].mean() * 100
    min_pct = tdf["position_pct"].min() * 100
    max_pct = tdf["position_pct"].max() * 100
    print(f"  {s:>10s}: avg={avg_pct:>5.1f}%  min={min_pct:>5.1f}%  max={max_pct:>5.1f}%")

# %% [markdown]
# ## 6. Analise trade-off retorno vs drawdown

# %%
print("\n" + "=" * 90)
print("ANALISE — Trade-off Retorno x Drawdown")
print("=" * 90)
print(f"{'Scheme':>11s}  {'Ret %':>8s}  {'|DD| %':>7s}  {'Ret/|DD|':>9s}  {'Sharpe':>7s}")
print("-" * 90)
ranked = []
for s in schemes:
    r = results[s]
    abs_dd = abs(r["max_dd_pct"])
    rdd = r["ret_pct"] / abs_dd if abs_dd > 1e-9 else 0.0
    ranked.append((s, r["ret_pct"], abs_dd, rdd, r["sharpe"]))
    print(f"{s:>11s}  {r['ret_pct']:>+7.2f}  {abs_dd:>6.2f}  {rdd:>+8.2f}  {r['sharpe']:>+7.2f}")

print("\nMelhor por metrica:")
best_ret = max(ranked, key=lambda x: x[1])
best_dd = min(ranked, key=lambda x: x[2])  # menor |DD|
best_rdd = max(ranked, key=lambda x: x[3])
best_sh = max(ranked, key=lambda x: x[4])
print(f"  Maior retorno:        {best_ret[0]} ({best_ret[1]:+.2f}%)")
print(f"  Menor drawdown:       {best_dd[0]} ({best_dd[2]:.2f}%)")
print(f"  Melhor Ret/|DD|:      {best_rdd[0]} ({best_rdd[3]:+.2f})")
print(f"  Melhor Sharpe:        {best_sh[0]} ({best_sh[4]:+.2f})")

# %% [markdown]
# ## 7. Recomendacao

# %%
print("\n" + "=" * 90)
print("RECOMENDACAO")
print("=" * 90)
full = results["FULL"]
print(
    f"FULL (producao atual): ret={full['ret_pct']:+.2f}%  maxDD={full['max_dd_pct']:+.2f}%  "
    f"sharpe={full['sharpe']:+.2f}"
)
print(
    "  >> RISCO: FULL aloca 100% do capital por trade. Em caso de stop -3*ATR "
    "(~-9% a -12% do entry com ATR de 3-4%), o drawdown por trade individual "
    "ja consome ~10% do capital. Sequencias de stops podem causar drawdowns severos."
)
print()
print("Trade-offs:")
print("  - FULL maximiza retorno esperado, mas com volatilidade total da estrategia")
print("  - HALF/QUARTER reduzem variancia ~linearmente; retorno cai na mesma proporcao")
print("  - RISK-1PCT normaliza risco por trade (stop = -1% capital) — drawdown previsivel")
print("  - KELLY_FRAC e ultra-conservador (2.5%) — capital praticamente nao move")

# Sugestao por ranking
print("\nSugestao final:")
print(f"  - Se prioridade = retorno absoluto:    {best_ret[0]}")
print(f"  - Se prioridade = sobrevivencia/DD:    RISK-1PCT (DD calibrado por trade)")
print(f"  - Se prioridade = risk-adjusted:       {best_sh[0]} (melhor Sharpe)")
print(f"  - Se prioridade = Ret/|DD|:            {best_rdd[0]}")
