"""exp_reality_sim — Replay histórico como se bot estivesse ao vivo.

Emite cada trade fechado em formato Telegram (entrada, saída, PnL), simulando
o que VOCÊ teria visto no celular durante 2023-2026.

Usa probas cached (data/walk_forward_probas.parquet) + simulação completa
(position blocking, compounding, custos, sem_BEAR filter).

Reporta:
- Lista completa de trades em formato readable
- Mês a mês: trades + PnL
- Equity curve mensal (texto)
- Maior win, maior loss, maior streak
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

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
THR = 0.35
NO_BEAR = -0.05
BARS_PER_MONTH = 180
INITIAL = 1000.0


def build_mat():
    df = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME, lag=1).drop_nulls(subset=["atr_14"])
    lab_df = lab.triple_barrier(df, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=HORIZON)
    m = lab_df.select(["open_time", "close", "high", "low", "atr_14", "barrier_ret"]).to_pandas()
    m["dt"] = m["open_time"].apply(lambda ms: datetime.fromtimestamp(ms/1000, tz=timezone.utc))
    return m


probas_df = pd.read_parquet(ROOT / "data" / "walk_forward_probas.parquet")
mat = build_mat()
merged = mat.merge(probas_df[["open_time", "proba_mid"]], on="open_time", how="left")
merged["proba_mid"] = merged["proba_mid"].fillna(0)
covered = merged["proba_mid"] > 0

# Simulação realista
capital = INITIAL
cash = capital
position = None
trades = []

print("=" * 90)
print(f" REPLAY HISTÓRICO — como se bot estivesse ao vivo de {merged.loc[covered, 'dt'].min():%Y-%m-%d}")
print("=" * 90)
print(f" Config: thr={THR}, no_bear={NO_BEAR}, MID-only, FULL sizing, COST={COST*100:.2f}%")
print(f" Capital inicial: ${INITIAL:,.0f}")
print("=" * 90)

for i, row in merged.iterrows():
    if not covered.iloc[i]:
        continue
    close = row["close"]; high = row["high"]; low = row["low"]
    atr = row["atr_14"]; ot = row["open_time"]; dt = row["dt"]
    proba = row["proba_mid"]

    # 1) Verifica posição aberta
    if position is not None:
        hit_stop = low <= position["stop"]
        hit_target = high >= position["target"]
        timeout = ot >= position["timeout_at"]
        exit_p, outcome, emoji = None, None, None
        if hit_stop:
            exit_p, outcome, emoji = position["stop"], "STOP", "🔴"
        elif hit_target:
            exit_p, outcome, emoji = position["target"], "TARGET", "🟢"
        elif timeout:
            exit_p, outcome, emoji = close, "TIMEOUT", "⏱️"
        if exit_p is not None:
            pnl_pct = (exit_p / position["entry"] - 1) - COST
            pnl_usd = position["size_usd"] * pnl_pct
            cash += position["size_usd"] + pnl_usd
            capital = cash
            duration_h = (ot - position["open_ot"]) / (1000 * 3600)
            print(f"\n{emoji} {outcome}  {dt:%Y-%m-%d %H:%M}  exit ${exit_p:,.0f}  "
                  f"PnL {pnl_pct*100:+.2f}%  ({duration_h:.0f}h)  → capital ${capital:,.0f}")
            trades.append({
                "entry_dt": position["entry_dt"], "exit_dt": dt,
                "entry": position["entry"], "exit": exit_p,
                "pnl_pct": pnl_pct, "outcome": outcome,
                "capital_after": capital, "duration_h": duration_h,
            })
            position = None

    # 2) Avalia novo sinal
    if position is None and proba > THR:
        # bear filter
        if i >= BARS_PER_MONTH:
            ret_30d = close / merged["close"].iloc[i - BARS_PER_MONTH] - 1
            if ret_30d < NO_BEAR:
                continue
        entry = close
        stop = entry - ATR_MULT * atr
        target = entry + ATR_MULT * atr
        size_usd = capital
        cash -= size_usd
        position = {
            "entry": entry, "stop": stop, "target": target,
            "size_usd": size_usd, "open_ot": ot, "entry_dt": dt,
            "timeout_at": ot + HORIZON * 4 * 3600 * 1000,
        }
        print(f"\n🟢 SINAL DE COMPRA  {dt:%Y-%m-%d %H:%M}  entry ${entry:,.0f}  "
              f"target ${target:,.0f} ({(target/entry-1)*100:+.2f}%)  "
              f"stop ${stop:,.0f} ({(stop/entry-1)*100:+.2f}%)  proba {100*proba:.1f}%")

# Fecha posição aberta no final (se houver)
if position is not None:
    last = merged.iloc[-1]
    pnl_pct = (last["close"] / position["entry"] - 1) - COST
    cash += position["size_usd"] * (1 + pnl_pct)
    capital = cash
    trades.append({"entry_dt": position["entry_dt"], "exit_dt": last["dt"],
                   "entry": position["entry"], "exit": last["close"],
                   "pnl_pct": pnl_pct, "outcome": "FORCED", "capital_after": capital,
                   "duration_h": (last["open_time"] - position["open_ot"]) / 3600000})

# Sumário final
print("\n" + "=" * 90)
print(" SUMÁRIO DA SIMULAÇÃO")
print("=" * 90)
tr = pd.DataFrame(trades)
print(f"\nCapital inicial: ${INITIAL:,.0f}")
print(f"Capital final:   ${capital:,.0f}")
print(f"Retorno total:   {100*(capital/INITIAL - 1):+.1f}%")
print(f"Trades fechados: {len(tr)}")
print(f"  🟢 TARGET:  {(tr['outcome']=='TARGET').sum()}")
print(f"  🔴 STOP:    {(tr['outcome']=='STOP').sum()}")
print(f"  ⏱️ TIMEOUT: {(tr['outcome']=='TIMEOUT').sum()}")
print(f"\nWin rate (PnL>0):  {100*(tr['pnl_pct'] > 0).mean():.1f}%")
print(f"Avg PnL/trade:     {100*tr['pnl_pct'].mean():+.3f}%")
print(f"Melhor trade:      {100*tr['pnl_pct'].max():+.2f}%  ({tr.loc[tr['pnl_pct'].idxmax(), 'exit_dt'].date()})")
print(f"Pior trade:        {100*tr['pnl_pct'].min():+.2f}%  ({tr.loc[tr['pnl_pct'].idxmin(), 'exit_dt'].date()})")
print(f"Duração média:     {tr['duration_h'].mean():.1f}h")

# Streaks
streaks = []
cur = 0; cur_type = None
for pnl in tr["pnl_pct"]:
    t = "W" if pnl > 0 else "L"
    if t == cur_type:
        cur += 1
    else:
        if cur > 0:
            streaks.append((cur_type, cur))
        cur = 1; cur_type = t
if cur > 0:
    streaks.append((cur_type, cur))
win_streaks = [s[1] for s in streaks if s[0] == "W"]
loss_streaks = [s[1] for s in streaks if s[0] == "L"]
print(f"\nMaior win streak:  {max(win_streaks) if win_streaks else 0}")
print(f"Maior loss streak: {max(loss_streaks) if loss_streaks else 0}")

# Mensal
print(f"\n{'='*60}")
print(" PERFORMANCE MENSAL")
print(f"{'='*60}")
tr["month"] = tr["exit_dt"].dt.to_period("M")
monthly = tr.groupby("month").agg(
    n=("pnl_pct", "size"),
    win_rate=("pnl_pct", lambda x: (x > 0).mean()),
    total_pnl=("pnl_pct", "sum"),
    capital_end=("capital_after", "last"),
).reset_index()

print(f"\n{'mês':<10s}  {'n':>4s}  {'win%':>6s}  {'sum PnL':>9s}  {'capital fim':>13s}")
print("-" * 55)
for _, r in monthly.iterrows():
    icon = "✓" if r["total_pnl"] > 0 else "✗"
    print(f"  {str(r['month']):<8s}  {int(r['n']):>4d}  {100*r['win_rate']:>5.1f}%  {100*r['total_pnl']:>+7.1f}%  ${r['capital_end']:>10,.0f}  {icon}")

# Comparação B&H
bh_entry = merged.loc[covered, "close"].iloc[0]
bh_final = merged["close"].iloc[-1]
bh_cap = INITIAL * (bh_final / bh_entry)
print(f"\n{'='*60}")
print(f" vs Buy-and-Hold mesmo período: ${bh_cap:,.0f} ({(bh_cap/INITIAL-1)*100:+.1f}%)")
print(f" Modelo:                        ${capital:,.0f} ({(capital/INITIAL-1)*100:+.1f}%)")
print(f" Diferença:                     ${capital - bh_cap:+,.0f}")
print(f"{'='*60}")
