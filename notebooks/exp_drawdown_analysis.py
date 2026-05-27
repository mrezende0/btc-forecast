# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # EXP — Drawdown & Estabilidade Psicologica (DUAL-HORIZON AND)
#
# Reproduz o sistema de PRODUCAO (mid h=12 AND long h=18, threshold 0.35)
# via walk-forward quarterly expanding (>=2023). Para cada trade alinhado por
# open_time aplica COST=0.08% e calcula equity curve com capital base $1000
# (full-size por trade).
#
# Metricas:
#   1.  Max drawdown ($ e %)
#   2.  Maior streak de losses consecutivas
#   3.  Maior streak de wins consecutivas
#   4.  Tempo em underwater (% do tempo onde equity < previous peak)
#   5.  Tempo medio para recuperar de drawdown (TTR)
#   6.  Pior trimestre / mes / semana
#   7.  Distribuicao de PnL por trade (histograma textual)
#   8.  Calmar ratio (CAGR / MaxDD)
#   9.  Sortino ratio
#   10. Identificacao do pior periodo + analise narrativa

# %%
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from pipeline import features as feat, labels as lab

# ---------------- CONFIG ----------------
TIMEFRAME = 240
HORIZON_MID = 12
HORIZON_LONG = 18
ATR_MULT = 3.0
COST = 0.0008
THRESHOLD = 0.35
CAPITAL_BASE = 1000.0

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
# ## 1. Build matrix + labels para os 2 horizontes

# %%
print("Building feature matrix @ 4h...")
df_base = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME, lag=1).drop_nulls(subset=["atr_14"])
print(f"  shape: {df_base.shape}")

mats: dict[str, pd.DataFrame] = {}
for name, h in [("mid", HORIZON_MID), ("long", HORIZON_LONG)]:
    lbl = lab.triple_barrier(df_base, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=h)
    lbl = lbl.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y_bin"))
    if name == "mid":
        feature_cols = [
            c for c in lbl.columns
            if c not in feat.LAG_SAFE_EXCLUDE
            and c not in {"label", "hit_bar", "barrier_ret", "upper_px", "lower_px", "y_bin"}
        ]
    m = lbl.select(["open_time", "close", "y_bin", "barrier_ret", *feature_cols]).drop_nulls(
        subset=feature_cols + ["y_bin"]
    ).to_pandas()
    m["dt"] = m["open_time"].apply(lambda ms: datetime.fromtimestamp(ms / 1000, tz=timezone.utc))
    m["quarter"] = m["dt"].dt.to_period("Q")
    mats[name] = m
    print(f"  {name:>5s} (h={h}): {len(m)} rows  {m['dt'].min().date()} -> {m['dt'].max().date()}")

# %% [markdown]
# ## 2. Walk-forward quarterly expanding -> probas por horizonte

# %%
quarters = [q for q in sorted(mats["mid"]["quarter"].unique()) if q.start_time.year >= 2023]
print(f"\nQuarters de teste: {len(quarters)} ({quarters[0]} -> {quarters[-1]})")

proba_by_h: dict[str, dict[int, float]] = {"mid": {}, "long": {}}

for q in quarters:
    for name, h in [("mid", HORIZON_MID), ("long", HORIZON_LONG)]:
        m = mats[name]
        test_idx = m.index[m["quarter"] == q].tolist()
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

        model = lgb.train(PARAMS, lgb.Dataset(X_tr, y_tr), num_boost_round=N_ROUNDS)
        proba = model.predict(X_te)
        for ot, p in zip(ot_te, proba):
            proba_by_h[name][int(ot)] = float(p)
    print(f"  done {q}")

# %% [markdown]
# ## 3. Constroi lista de TRADES do dual-horizon AND (alinhado por open_time)

# %%
ots_common = sorted(set(proba_by_h["mid"].keys()) & set(proba_by_h["long"].keys()))
print(f"\nOpen_times comuns: {len(ots_common)}")

p_mid = np.array([proba_by_h["mid"][o] for o in ots_common])
p_long = np.array([proba_by_h["long"][o] for o in ots_common])

# PnL eh com ret do mid (horizonte realizado mais curto = trade fecha em ate 48h)
ret_lut = dict(zip(mats["mid"]["open_time"].astype(int), mats["mid"]["barrier_ret"]))
ret_all = np.array([ret_lut[int(o)] for o in ots_common])

signal = (p_mid > THRESHOLD) & (p_long > THRESHOLD)
trade_ots = np.array(ots_common)[signal]
trade_rets = ret_all[signal] - COST
trade_dts = pd.to_datetime(trade_ots, unit="ms", utc=True)

trades = pd.DataFrame({"dt": trade_dts, "ret_net": trade_rets}).sort_values("dt").reset_index(drop=True)
print(f"Trades dual-horizon AND: {len(trades)}")
print(f"Range trades: {trades['dt'].min().date()} -> {trades['dt'].max().date()}")

# %% [markdown]
# ## 4. Equity curve (capital base $1000, full size, compound)

# %%
trades["pnl_usd"] = CAPITAL_BASE * trades["ret_net"]  # full-size sempre $1000 (simple, nao compound)
# Modo compound: cada trade re-investe equity atual
equity_compound = [CAPITAL_BASE]
for r in trades["ret_net"]:
    equity_compound.append(equity_compound[-1] * (1 + r))
trades["equity_compound"] = equity_compound[1:]

# Modo nao-compound (sempre $1000 base por trade)
trades["equity_simple"] = CAPITAL_BASE + trades["pnl_usd"].cumsum()

# Usamos compound como principal (mais realista pra retail; reinveste lucros)
trades["equity"] = trades["equity_compound"]
trades["peak"] = trades["equity"].cummax()
trades["dd_usd"] = trades["equity"] - trades["peak"]
trades["dd_pct"] = trades["dd_usd"] / trades["peak"]

# %% [markdown]
# ## 5. Streaks e tempo underwater

# %%
def max_streak(arr_bool: np.ndarray) -> tuple[int, int, int]:
    """Retorna (max_len, start_idx, end_idx) do maior run de True."""
    if len(arr_bool) == 0:
        return 0, -1, -1
    best, best_s, best_e = 0, -1, -1
    cur, cur_s = 0, 0
    for i, v in enumerate(arr_bool):
        if v:
            if cur == 0:
                cur_s = i
            cur += 1
            if cur > best:
                best, best_s, best_e = cur, cur_s, i
        else:
            cur = 0
    return best, best_s, best_e


wins = trades["ret_net"].values > 0
losses = ~wins

max_loss_streak, ls_s, ls_e = max_streak(losses)
max_win_streak, ws_s, ws_e = max_streak(wins)

# Tempo underwater: precisamos olhar EQUITY CURVE no eixo temporal (nao trade-count)
# Resample diario do equity (forward-fill)
eq_daily = trades.set_index("dt")["equity"].resample("1D").last().ffill()
# Prepend baseline ate primeiro trade nao eh necessario - comeca no primeiro trade
peak_d = eq_daily.cummax()
underwater_d = eq_daily < peak_d
pct_underwater = float(underwater_d.mean()) * 100

# Tempo medio para recuperar (TTR): para cada drawdown completo, dias entre peak e novo high
ttr_days: list[int] = []
in_dd = False
dd_start = None
for ts, uw in underwater_d.items():
    if uw and not in_dd:
        in_dd = True
        dd_start = ts
    elif not uw and in_dd:
        in_dd = False
        ttr_days.append((ts - dd_start).days)
avg_ttr = float(np.mean(ttr_days)) if ttr_days else 0.0
max_ttr = int(np.max(ttr_days)) if ttr_days else 0

# %% [markdown]
# ## 6. Pior periodo (trimestre / mes / semana) por PnL

# %%
trades["quarter"] = trades["dt"].dt.to_period("Q")
trades["month"] = trades["dt"].dt.to_period("M")
trades["week"] = trades["dt"].dt.to_period("W")

pnl_by_q = trades.groupby("quarter")["ret_net"].agg(["sum", "count", "mean"])
pnl_by_m = trades.groupby("month")["ret_net"].agg(["sum", "count", "mean"])
pnl_by_w = trades.groupby("week")["ret_net"].agg(["sum", "count", "mean"])

worst_q = pnl_by_q["sum"].idxmin()
worst_m = pnl_by_m["sum"].idxmin()
worst_w = pnl_by_w["sum"].idxmin()

# %% [markdown]
# ## 7. Calmar e Sortino

# %%
n_years = (trades["dt"].max() - trades["dt"].min()).total_seconds() / (365.25 * 86400)
final_eq = trades["equity"].iloc[-1]
total_ret = final_eq / CAPITAL_BASE - 1
cagr = (final_eq / CAPITAL_BASE) ** (1 / n_years) - 1 if n_years > 0 else 0.0

max_dd_pct = float(trades["dd_pct"].min())  # negativo
max_dd_usd = float(trades["dd_usd"].min())
calmar = cagr / abs(max_dd_pct) if max_dd_pct != 0 else float("nan")

# Sortino: usa retornos diarios (resampled) com risk-free=0
ret_daily = eq_daily.pct_change().dropna()
downside = ret_daily[ret_daily < 0]
downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
mean_daily = float(ret_daily.mean())
sortino = (mean_daily / downside_std * np.sqrt(365)) if downside_std > 0 else float("nan")

# Sharpe tradicional para referencia
sharpe_daily = (mean_daily / ret_daily.std(ddof=1) * np.sqrt(365)) if ret_daily.std(ddof=1) > 0 else float("nan")

# %% [markdown]
# ## 8. Histograma textual de PnL por trade

# %%
def text_hist(values: np.ndarray, bins: list[float], labels: list[str]) -> str:
    """Histograma textual com barras horizontais."""
    counts = np.zeros(len(labels), dtype=int)
    for v in values:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                counts[i] += 1
                break
        else:
            if v >= bins[-1]:
                counts[-1] += 1
    max_c = counts.max() if counts.max() > 0 else 1
    width = 40
    lines = []
    for lbl, c in zip(labels, counts):
        bar = "#" * int(width * c / max_c)
        pct = 100 * c / len(values) if len(values) else 0
        lines.append(f"  {lbl:>14s} | {bar:<40s} {c:>4d} ({pct:>4.1f}%)")
    return "\n".join(lines)


bins = [-1.0, -0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05, 0.10, 1.0]
labels_h = [
    "<-5%", "-5% a -3%", "-3% a -1%", "-1% a 0%",
    "0% a 1%", "1% a 3%", "3% a 5%", "5% a 10%", ">=10%",
]

# %% [markdown]
# ## 9. Relatorio final

# %%
print("\n" + "=" * 80)
print("RELATORIO DE DRAWDOWN E ESTABILIDADE PSICOLOGICA - DUAL-HORIZON AND")
print("=" * 80)
print(f"Capital inicial: ${CAPITAL_BASE:.2f}  |  Custo: {COST*100:.2f}%  |  Threshold: {THRESHOLD}")
print(f"Periodo:         {trades['dt'].min().date()} -> {trades['dt'].max().date()}  ({n_years:.2f} anos)")
print(f"N trades:        {len(trades)}  ({len(trades)/n_years:.1f}/ano)")
print()

print("=" * 80)
print("METRICAS PRINCIPAIS")
print("=" * 80)
hdr = f"{'metrica':<40s} | {'valor':>20s}"
print(hdr)
print("-" * len(hdr))

n_wins = int(wins.sum())
n_losses = int(losses.sum())
win_rate = n_wins / len(trades) if len(trades) else 0
avg_win = trades.loc[wins, "ret_net"].mean() if n_wins else 0
avg_loss = trades.loc[losses, "ret_net"].mean() if n_losses else 0
profit_factor = abs(trades.loc[wins, "ret_net"].sum() / trades.loc[losses, "ret_net"].sum()) if n_losses else float("inf")

rows = [
    ("Equity final",                       f"${final_eq:>10.2f}"),
    ("Retorno total",                      f"{total_ret*100:>+10.2f}%"),
    ("CAGR",                                f"{cagr*100:>+10.2f}%"),
    ("Win rate",                            f"{win_rate*100:>10.1f}%"),
    ("Trade medio (win)",                   f"{avg_win*100:>+10.2f}%"),
    ("Trade medio (loss)",                  f"{avg_loss*100:>+10.2f}%"),
    ("Profit factor",                       f"{profit_factor:>10.2f}"),
    ("",                                    ""),
    ("Max Drawdown (%)",                    f"{max_dd_pct*100:>+10.2f}%"),
    ("Max Drawdown ($)",                    f"${max_dd_usd:>+10.2f}"),
    ("Max streak losses consecutivas",      f"{max_loss_streak:>10d}"),
    ("Max streak wins consecutivas",        f"{max_win_streak:>10d}"),
    ("Tempo underwater (% do periodo)",     f"{pct_underwater:>10.1f}%"),
    ("TTR medio (dias)",                    f"{avg_ttr:>10.1f}"),
    ("TTR maximo (dias)",                   f"{max_ttr:>10d}"),
    ("",                                    ""),
    ("Calmar ratio (CAGR/|MaxDD|)",         f"{calmar:>10.2f}"),
    ("Sortino ratio (anualizado)",          f"{sortino:>10.2f}"),
    ("Sharpe ratio (anualizado, daily)",    f"{sharpe_daily:>10.2f}"),
]
for k, v in rows:
    print(f"{k:<40s} | {v:>20s}")

# Pior periodo
print()
print("=" * 80)
print("PIORES PERIODOS")
print("=" * 80)
print(f"Pior TRIMESTRE: {worst_q}  -> PnL={100*pnl_by_q.loc[worst_q,'sum']:+.2f}%  ({int(pnl_by_q.loc[worst_q,'count'])} trades)")
print(f"Pior MES:       {worst_m}  -> PnL={100*pnl_by_m.loc[worst_m,'sum']:+.2f}%  ({int(pnl_by_m.loc[worst_m,'count'])} trades)")
print(f"Pior SEMANA:    {worst_w}  -> PnL={100*pnl_by_w.loc[worst_w,'sum']:+.2f}%  ({int(pnl_by_w.loc[worst_w,'count'])} trades)")

# Streak de losses - datas
if max_loss_streak > 0:
    print()
    print(f"Maior sequencia de LOSSES ({max_loss_streak} consecutivas):")
    print(f"  {trades['dt'].iloc[ls_s].date()} -> {trades['dt'].iloc[ls_e].date()}")
    print(f"  PnL acumulado no streak: {100*trades['ret_net'].iloc[ls_s:ls_e+1].sum():+.2f}%")
if max_win_streak > 0:
    print(f"\nMaior sequencia de WINS ({max_win_streak} consecutivas):")
    print(f"  {trades['dt'].iloc[ws_s].date()} -> {trades['dt'].iloc[ws_e].date()}")
    print(f"  PnL acumulado no streak: {100*trades['ret_net'].iloc[ws_s:ws_e+1].sum():+.2f}%")

# Identifica DD maximo - datas exatas
dd_trough_idx = int(trades["dd_pct"].idxmin())
# Peak anterior
peak_before = trades.loc[:dd_trough_idx, "peak"].iloc[-1]
peak_idx = int(trades.loc[:dd_trough_idx, "equity"].idxmax())
# Recuperacao (se houver)
post = trades.iloc[dd_trough_idx:]
recover = post[post["equity"] >= peak_before]
recover_idx = int(recover.index[0]) if len(recover) else None

print()
print("=" * 80)
print("ANATOMIA DO MAX DRAWDOWN")
print("=" * 80)
print(f"Pico:        idx={peak_idx:>4d}  {trades['dt'].iloc[peak_idx].date()}  equity=${trades['equity'].iloc[peak_idx]:.2f}")
print(f"Fundo:       idx={dd_trough_idx:>4d}  {trades['dt'].iloc[dd_trough_idx].date()}  equity=${trades['equity'].iloc[dd_trough_idx]:.2f}")
print(f"Profundidade: {max_dd_pct*100:+.2f}%  (${max_dd_usd:+.2f})")
print(f"Duracao pico->fundo: {(trades['dt'].iloc[dd_trough_idx] - trades['dt'].iloc[peak_idx]).days} dias")
print(f"N trades no DD: {dd_trough_idx - peak_idx}")
if recover_idx is not None:
    print(f"Recuperacao: idx={recover_idx}  {trades['dt'].iloc[recover_idx].date()}  ({(trades['dt'].iloc[recover_idx] - trades['dt'].iloc[peak_idx]).days} dias do pico)")
else:
    print("Recuperacao: NAO RECUPEROU ate o fim da serie")

# Histograma
print()
print("=" * 80)
print("DISTRIBUICAO DE PnL POR TRADE")
print("=" * 80)
print(text_hist(trades["ret_net"].values, bins, labels_h))

# Quarter table
print()
print("=" * 80)
print("PnL POR TRIMESTRE")
print("=" * 80)
print(f"{'quarter':>8s} | {'n':>4s} | {'PnL%':>8s} | {'avg%':>8s} | {'sinal':>30s}")
print("-" * 75)
for q, row in pnl_by_q.iterrows():
    pnl_q = 100 * row["sum"]
    avg_q = 100 * row["mean"]
    n_q = int(row["count"])
    bar_w = int(min(30, abs(pnl_q)))
    bar = ("+" * bar_w) if pnl_q >= 0 else ("-" * bar_w)
    print(f"{str(q):>8s} | {n_q:>4d} | {pnl_q:>+7.2f}% | {avg_q:>+7.3f}% | {bar:>30s}")

# %% [markdown]
# ## 10. Veredicto psicologico (heuristico)

# %%
print()
print("=" * 80)
print("VEREDICTO PSICOLOGICO")
print("=" * 80)

flags = []
if abs(max_dd_pct) > 0.50:
    flags.append(f"[CRITICO] MaxDD {max_dd_pct*100:.1f}% > 50% - quase ninguem aguenta operar")
elif abs(max_dd_pct) > 0.30:
    flags.append(f"[ALERTA] MaxDD {max_dd_pct*100:.1f}% > 30% - doloroso para retail")
else:
    flags.append(f"[OK] MaxDD {max_dd_pct*100:.1f}% - dentro de zona suportavel")

if max_loss_streak > 7:
    flags.append(f"[ALERTA] Streak {max_loss_streak} losses > 7 - investidor vai achar que 'modelo quebrou'")
else:
    flags.append(f"[OK] Streak max de losses = {max_loss_streak} (<= 7)")

if pct_underwater > 70:
    flags.append(f"[ALERTA] {pct_underwater:.0f}% do tempo underwater - sensacao constante de prejuizo")
elif pct_underwater > 50:
    flags.append(f"[ATENCAO] {pct_underwater:.0f}% do tempo underwater - acima da metade")
else:
    flags.append(f"[OK] {pct_underwater:.0f}% do tempo underwater")

if not np.isnan(calmar):
    if calmar > 1.0:
        flags.append(f"[OK] Calmar {calmar:.2f} > 1.0 - retorno compensa o DD")
    elif calmar > 0.5:
        flags.append(f"[ATENCAO] Calmar {calmar:.2f} entre 0.5-1.0 - marginal")
    else:
        flags.append(f"[ALERTA] Calmar {calmar:.2f} < 0.5 - DD muito grande para o retorno")

if not np.isnan(sortino):
    if sortino > 2:
        flags.append(f"[OK] Sortino {sortino:.2f} > 2 - downside controlado")
    elif sortino > 1:
        flags.append(f"[ATENCAO] Sortino {sortino:.2f} entre 1-2")
    else:
        flags.append(f"[ALERTA] Sortino {sortino:.2f} < 1 - downside excessivo")

for f in flags:
    print(f"  {f}")

print()
print("OPERAVEL PARA RETAIL?")
critical = sum(1 for f in flags if "[CRITICO]" in f)
alerts = sum(1 for f in flags if "[ALERTA]" in f)
if critical > 0:
    veredito = "NAO - precisa protecao adicional URGENTE (stop loss agregado, sizing reduzido)"
elif alerts >= 2:
    veredito = "TALVEZ - operavel com sizing reduzido (0.5x) e regra de pause apos N losses"
elif alerts == 1:
    veredito = "SIM com ressalvas - operavel mas precisa monitorar metricas-chave"
else:
    veredito = "SIM - perfil estavel para retail sem stress excessivo"
print(f"  {veredito}")

print()
print("RECOMENDACOES DE PROTECAO ADICIONAL:")
recs = []
if abs(max_dd_pct) > 0.25:
    recs.append("- Sizing fracionario (0.3-0.5x equity) ao inves de full-size para suavizar DD")
if max_loss_streak >= 5:
    recs.append(f"- Circuit breaker: pausar trades apos {max_loss_streak} losses consecutivas (revisar dados)")
if pct_underwater > 50:
    recs.append("- Trailing stop no equity: reduzir size quando equity < 90% peak")
if not np.isnan(calmar) and calmar < 1.0:
    recs.append("- Filtro de regime (EMA200, VIX) para evitar trades em mercado adverso")
if max_ttr > 180:
    recs.append(f"- TTR max {max_ttr}d > 6 meses: considerar diversificacao com short model")
if not recs:
    recs.append("- Sistema robusto. Manter monitoramento mensal de metricas.")
for r in recs:
    print(f"  {r}")

print()
print("=" * 80)
print("FIM DO RELATORIO")
print("=" * 80)
