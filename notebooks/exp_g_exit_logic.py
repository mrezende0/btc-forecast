"""exp_g_exit_logic — Backtest contrafactual de saída refinada.

Compara 6 modos no cache walk_forward_probas.parquet (winner A1-A):
  1. baseline             — hard target 3×ATR, hard stop 3×ATR, timeout 48h (prod hoje)
  2. partial_tp_50        — 50% sai em +1×ATR, 50% restante com trailing chandelier (high - 2×ATR)
  3. partial_tp_30_30_40  — 30% em +1×ATR, 30% em +2×ATR, 40% trailing remainder
  4. trail_only           — 100% trailing chandelier desde início (sem hard target)
  5. tp1_be               — 50% sai em +1×ATR, 50% restante: stop move pra break-even, target full 3×ATR
  6. tp1_be_wide_trail    — 50% sai em +1×ATR, 50% restante com trailing FROUXO (high - 4×ATR)

Trailing chandelier: stop_t = max(stop_inicial, max(high até t) - K×ATR_entry).
                     Stop só sobe, nunca desce.

Sem K extra: aplica lógica de saída sobre probas existentes.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HORIZON_MID = 12
ATR_MULT = 3.0
TRAIL_ATR_MULT = 2.0    # chandelier: high - 2×ATR
COST = 0.0015           # custo round-trip aplicado em EACH partial exit proporcional ao tamanho
THR = 0.35
NO_BEAR = -0.05
INITIAL = 1000.0
BARS_PER_YEAR = 6 * 365

VAL_END = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
HOLDOUT_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
TRAIN_START = datetime(2023, 1, 1, tzinfo=timezone.utc)

PROBAS_CACHE = ROOT / "data" / "walk_forward_probas.parquet"


def simulate(probas: pd.DataFrame, mode: str) -> dict:
    """Simulação intra-bar honesta:
    - cada vela tem high/low/close
    - ordem de checagem em conflito: stop antes do target (conservador)
    - partial TP cria múltiplos exits no mesmo trade
    """
    closes = probas["close"].to_numpy()
    highs = probas["high"].to_numpy()
    lows = probas["low"].to_numpy()
    atrs = probas["atr"].to_numpy()
    pm = probas["proba_mid"].to_numpy()
    ret_30d = probas["ret_30d"].to_numpy()
    n = len(probas)

    capital = INITIAL
    equity = np.full(n, INITIAL)

    # estado da posição (escala 0-1 do notional inicial)
    in_pos = False
    entry_idx = -1
    entry_px = atr_entry = np.nan
    initial_stop = trail_stop = np.nan
    target_full = tp1 = tp2 = np.nan
    expiry_idx = -1
    fraction_open = 1.0     # fração restante (1.0 = trade inteiro aberto)
    highest_high = -np.inf  # pra trailing
    tp1_done = tp2_done = False
    trades = []
    legs = []   # cada perna de exit registrada

    for i in range(n):
        if in_pos:
            # atualiza highest pra trailing
            highest_high = max(highest_high, highs[i])
            # multiplicador do trailing varia por modo
            if mode == "tp1_be_wide_trail":
                trail_mult = 4.0
            else:
                trail_mult = TRAIL_ATR_MULT  # 2.0
            trail_stop_new = highest_high - trail_mult * atr_entry
            trail_stop = max(trail_stop, trail_stop_new)
            # effective_stop por modo
            if mode in ("partial_tp_50", "partial_tp_30_30_40", "trail_only"):
                effective_stop = trail_stop
            elif mode == "tp1_be_wide_trail":
                # depois do tp1, usa trailing frouxo; antes, usa initial_stop
                effective_stop = trail_stop if tp1_done else initial_stop
            elif mode == "tp1_be":
                # depois do tp1, stop move pra break-even (entry); antes, initial_stop
                effective_stop = max(initial_stop, entry_px) if tp1_done else initial_stop
            else:
                effective_stop = initial_stop

            # 1) stop (intra-bar low fura): fecha tudo que resta
            if lows[i] <= effective_stop:
                px_ret = effective_stop / entry_px - 1
                net = fraction_open * px_ret - fraction_open * COST
                capital *= (1 + net)
                legs.append({"trade_id": entry_idx, "exit_idx": i, "frac": fraction_open,
                             "px_ret": px_ret, "kind": "stop"})
                trades.append({"entry_idx": entry_idx, "exit_idx": i, "kind": "stop_closed",
                               "n_legs": len([l for l in legs if l['trade_id'] == entry_idx])})
                in_pos = False
                fraction_open = 0.0
                equity[i] = capital
                continue

            # 2) partial TP checks (em ordem: tp1, tp2, target_full)
            if mode in ("partial_tp_50", "tp1_be", "tp1_be_wide_trail") \
                    and not tp1_done and highs[i] >= tp1:
                frac = 0.5
                px_ret = tp1 / entry_px - 1
                net = frac * px_ret - frac * COST
                capital *= (1 + net)
                legs.append({"trade_id": entry_idx, "exit_idx": i, "frac": frac,
                             "px_ret": px_ret, "kind": "tp1"})
                fraction_open -= frac
                tp1_done = True

            elif mode == "partial_tp_30_30_40":
                if not tp1_done and highs[i] >= tp1:
                    frac = 0.30
                    px_ret = tp1 / entry_px - 1
                    net = frac * px_ret - frac * COST
                    capital *= (1 + net)
                    legs.append({"trade_id": entry_idx, "exit_idx": i, "frac": frac,
                                 "px_ret": px_ret, "kind": "tp1"})
                    fraction_open -= frac
                    tp1_done = True
                if not tp2_done and highs[i] >= tp2:
                    frac = 0.30
                    px_ret = tp2 / entry_px - 1
                    net = frac * px_ret - frac * COST
                    capital *= (1 + net)
                    legs.append({"trade_id": entry_idx, "exit_idx": i, "frac": frac,
                                 "px_ret": px_ret, "kind": "tp2"})
                    fraction_open -= frac
                    tp2_done = True

            # 3) target full (baseline E tp1_be — partial sai em tp1, resto vai até target_full)
            if mode in ("baseline", "tp1_be") and highs[i] >= target_full:
                px_ret = target_full / entry_px - 1
                net = fraction_open * px_ret - fraction_open * COST
                capital *= (1 + net)
                legs.append({"trade_id": entry_idx, "exit_idx": i, "frac": fraction_open,
                             "px_ret": px_ret, "kind": "target"})
                trades.append({"entry_idx": entry_idx, "exit_idx": i, "kind": "target_full"})
                in_pos = False
                fraction_open = 0.0
                equity[i] = capital
                continue

            # 4) timeout
            if i >= expiry_idx and fraction_open > 0:
                px_ret = closes[i] / entry_px - 1
                net = fraction_open * px_ret - fraction_open * COST
                capital *= (1 + net)
                legs.append({"trade_id": entry_idx, "exit_idx": i, "frac": fraction_open,
                             "px_ret": px_ret, "kind": "timeout"})
                trades.append({"entry_idx": entry_idx, "exit_idx": i, "kind": "timeout"})
                in_pos = False
                fraction_open = 0.0
                equity[i] = capital
                continue

            if fraction_open <= 1e-9:
                trades.append({"entry_idx": entry_idx, "exit_idx": i, "kind": "fully_partialed"})
                in_pos = False

        # entry
        if not in_pos and not (np.isnan(pm[i]) or np.isnan(atrs[i])):
            sig = pm[i] > THR
            if not np.isnan(ret_30d[i]) and ret_30d[i] < NO_BEAR:
                sig = False
            if sig:
                entry_idx = i
                entry_px = closes[i]
                atr_entry = atrs[i]
                initial_stop = entry_px - ATR_MULT * atr_entry
                trail_stop = initial_stop
                target_full = entry_px + ATR_MULT * atr_entry
                tp1 = entry_px + 1.0 * atr_entry
                tp2 = entry_px + 2.0 * atr_entry
                expiry_idx = i + HORIZON_MID
                in_pos = True
                fraction_open = 1.0
                highest_high = highs[i]
                tp1_done = tp2_done = False

        # mark-to-market equity
        if in_pos and fraction_open > 0:
            unreal = (closes[i] / entry_px - 1) * fraction_open
            equity[i] = capital * (1 + unreal - fraction_open * COST)
        else:
            equity[i] = capital

    return {"equity": equity, "trades": trades, "legs": legs}


def seg_metrics(eq: np.ndarray, dts: np.ndarray, seg: str) -> dict:
    if seg == "VAL":
        mask = (dts >= np.datetime64(TRAIN_START)) & (dts <= np.datetime64(VAL_END))
    elif seg == "HOLDOUT":
        mask = (dts >= np.datetime64(HOLDOUT_START))
    else:
        mask = np.ones(len(dts), dtype=bool)
    e = eq[mask]
    if len(e) < 30:
        return {"sharpe": np.nan, "max_dd": np.nan, "final": np.nan, "ret": np.nan}
    rets = np.diff(e) / e[:-1]
    rets = rets[~np.isnan(rets) & ~np.isinf(rets)]
    sharpe = float(rets.mean() / rets.std() * np.sqrt(BARS_PER_YEAR)) if rets.std() > 0 else 0.0
    peak = np.maximum.accumulate(e)
    max_dd = float((e / peak - 1).min())
    return {"sharpe": sharpe, "max_dd": max_dd, "final": float(INITIAL * e[-1] / e[0]),
            "ret": float(e[-1] / e[0] - 1)}


def main():
    if not PROBAS_CACHE.exists():
        raise SystemExit(f"cache ausente: {PROBAS_CACHE}")
    probas = pl.read_parquet(PROBAS_CACHE).to_pandas()
    probas["dt"] = pd.to_datetime(probas["dt"], utc=True)
    probas = probas[probas["proba_mid"].notna()].reset_index(drop=True)
    dts = probas["dt"].values

    modes = ["baseline", "partial_tp_50", "partial_tp_30_30_40", "trail_only",
             "tp1_be", "tp1_be_wide_trail"]
    print("=" * 120)
    print(" EXP-G — Exit Logic Backtest (cache A1-A)")
    print(f"  thr={THR}  no_bear={NO_BEAR}  ATR_MULT={ATR_MULT}  trail={TRAIL_ATR_MULT}×ATR  cost={COST}")
    print("=" * 120)
    header = f"{'mode':<24}{'VAL Shp':>10}{'HO Shp':>10}{'VAL DD':>10}{'HO DD':>10}{'VAL final':>14}{'HO final':>14}{'trades':>10}{'legs':>8}"
    print(header)
    print("-" * 120)

    results = {}
    for m in modes:
        sim = simulate(probas, m)
        val_m = seg_metrics(sim["equity"], dts, "VAL")
        ho_m = seg_metrics(sim["equity"], dts, "HOLDOUT")
        results[m] = (val_m, ho_m, sim)
        row = (
            f"{m:<24}"
            f"{val_m['sharpe']:>+10.3f}"
            f"{ho_m['sharpe']:>+10.3f}"
            f"{100*val_m['max_dd']:>+9.1f}%"
            f"{100*ho_m['max_dd']:>+9.1f}%"
            f"${val_m['final']:>12,.0f}"
            f"${ho_m['final']:>12,.0f}"
            f"{len(sim['trades']):>10d}"
            f"{len(sim['legs']):>8d}"
        )
        print(row)
    print("=" * 120)

    # breakdown legs por kind no winner
    print("\nBreakdown legs por kind (HOLDOUT):")
    ho_start = np.datetime64(HOLDOUT_START)
    for m in modes:
        legs_ho = [l for l in results[m][2]["legs"] if dts[l["exit_idx"]] >= ho_start]
        kinds = pd.Series([l["kind"] for l in legs_ho]).value_counts().to_dict()
        avg_ret = float(np.mean([l["px_ret"] * l["frac"] for l in legs_ho])) if legs_ho else 0.0
        print(f"  {m:<24} {dict(kinds)}  avg leg ret={100*avg_ret:+.3f}%")

    # Decisão
    base = results["baseline"]
    print("\nDECISÃO:")
    print(f"  Baseline HO Sharpe {base[1]['sharpe']:+.2f}  final ${base[1]['final']:,.0f}  DD {100*base[1]['max_dd']:+.1f}%")
    winners = []
    for m in ["partial_tp_50", "partial_tp_30_30_40", "trail_only", "tp1_be", "tp1_be_wide_trail"]:
        ho = results[m][1]
        d_sr = ho["sharpe"] - base[1]["sharpe"]
        d_final = ho["final"] - base[1]["final"]
        d_dd = ho["max_dd"] - base[1]["max_dd"]
        verdict = "KEEP" if (d_sr >= -0.1 and d_final > 0) or (d_sr > 0.15) else "KILL"
        print(f"  {m:<24}  ΔSharpe {d_sr:+.2f}  Δfinal ${d_final:+,.0f}  ΔDD {100*d_dd:+.1f}%  → {verdict}")
        if verdict == "KEEP":
            winners.append((m, ho))
    if winners:
        best = max(winners, key=lambda x: x[1]["sharpe"])
        print(f"\n  >>> WINNER: {best[0]}  (HO Sharpe {best[1]['sharpe']:+.2f}, final ${best[1]['final']:,.0f})")
    else:
        print("\n  >>> nenhum modo bate baseline. Ficar no hard target/stop atual.")


if __name__ == "__main__":
    main()
