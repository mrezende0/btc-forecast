"""exp_b_threshold_honest — Sweep HONESTO de threshold (MID-only, no_bear=-0.05).

Objetivo: usuário quer MAIS trades. Hoje em prod thr=0.35 dá 9 trades/mês.
Procurar sweet spot baixando thr sem destruir Sharpe.

Diferenças vs exp_threshold_grid (que tinha HIGH-3 bug):
- Ranking POR VAL Sharpe APENAS (VAL = 2023-01-01 → 2024-12-31, 2 anos)
- HOLDOUT (2025-01-01 → presente) só REPORTADO, NÃO escolhe
- Fixo: rule=MID, no_bear=-0.05, COST=0.0015, FULL sizing (100% notional)
- Grid 1D: thr ∈ {0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40, 0.425, 0.45}

Usa cache data/walk_forward_probas.parquet (probas A1-A já validadas).
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

# ----------------------------------------------------------------- params (lock = produção A1-A)
HORIZON_MID = 12        # 48h em 4h-bars
ATR_MULT = 3.0
COST = 0.0015
NO_BEAR = -0.05
RULE = "MID"
INITIAL_CAPITAL = 1000.0

VAL_END = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
HOLDOUT_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
TRAIN_START = datetime(2023, 1, 1, tzinfo=timezone.utc)  # bars antes disso = warmup do walk-forward

PROBAS_CACHE = ROOT / "data" / "walk_forward_probas.parquet"
OUT_CSV = ROOT / "data" / "exp_b_threshold_honest_results.csv"

THR_GRID = [0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40, 0.425, 0.45]


# ----------------------------------------------------------------- simulate (MID-only, FULL sizing)
def simulate(probas: pd.DataFrame, thr: float) -> dict:
    closes = probas["close"].to_numpy()
    highs = probas["high"].to_numpy()
    lows = probas["low"].to_numpy()
    atrs = probas["atr"].to_numpy()
    pm = probas["proba_mid"].to_numpy()
    ret_30d = probas["ret_30d"].to_numpy()
    n = len(probas)

    capital = INITIAL_CAPITAL
    in_pos = False
    entry_idx = -1
    entry_px = stop_px = target_px = np.nan
    expiry_idx = -1
    equity = np.full(n, INITIAL_CAPITAL)
    trades = []

    for i in range(n):
        # exit
        if in_pos:
            hit_target = highs[i] >= target_px
            hit_stop = lows[i] <= stop_px
            timed_out = i >= expiry_idx
            exit_now = False
            exit_px = np.nan
            if hit_stop:
                exit_px = stop_px
                exit_now = True
            elif hit_target:
                exit_px = target_px
                exit_now = True
            elif timed_out:
                exit_px = closes[i]
                exit_now = True
            if exit_now:
                net = exit_px / entry_px - 1 - COST
                capital *= (1 + net)
                trades.append({"entry_idx": entry_idx, "exit_idx": i, "net_ret": net})
                in_pos = False

        # entry (MID-only)
        if not in_pos and not (np.isnan(pm[i]) or np.isnan(atrs[i])):
            sig = pm[i] > thr
            if NO_BEAR is not None and not np.isnan(ret_30d[i]) and ret_30d[i] < NO_BEAR:
                sig = False
            if sig:
                entry_idx = i
                entry_px = closes[i]
                target_px = entry_px + ATR_MULT * atrs[i]
                stop_px = entry_px - ATR_MULT * atrs[i]
                expiry_idx = i + HORIZON_MID
                in_pos = True

        if in_pos:
            unreal = closes[i] / entry_px - 1
            equity[i] = capital * (1 + unreal - COST)
        else:
            equity[i] = capital

    # close final em mark-to-market (não fecha trade aberto pra não contaminar HO)
    return {"equity": equity, "n_trades": len(trades), "trades": trades}


def metrics(equity: np.ndarray, dts: np.ndarray, seg: str) -> dict:
    bars_per_year = 6 * 365
    if seg == "VAL":
        mask = (dts >= np.datetime64(TRAIN_START)) & (dts <= np.datetime64(VAL_END))
    elif seg == "HOLDOUT":
        mask = (dts >= np.datetime64(HOLDOUT_START))
    else:
        mask = np.ones(len(dts), dtype=bool)
    eq = equity[mask]
    if len(eq) < 30:
        return {"sharpe": np.nan, "psr_0": np.nan, "max_dd": np.nan, "ret": np.nan, "final": np.nan}
    rets = np.diff(eq) / eq[:-1]
    rets = rets[~np.isnan(rets)]
    if len(rets) < 30 or rets.std() == 0:
        return {"sharpe": 0.0, "psr_0": 0.5, "max_dd": 0.0, "ret": float(eq[-1] / eq[0] - 1),
                "final": float(INITIAL_CAPITAL * eq[-1] / eq[0])}
    sharpe = float(rets.mean() / rets.std() * np.sqrt(bars_per_year))
    from scipy.stats import norm
    sr = rets.mean() / rets.std()
    skew = float(pd.Series(rets).skew())
    kurt = float(pd.Series(rets).kurtosis())
    denom = np.sqrt(max(1e-12, (1 - skew * sr + kurt / 4 * sr ** 2) / (len(rets) - 1)))
    psr_0 = float(norm.cdf(sr / denom))
    peak = np.maximum.accumulate(eq)
    max_dd = float((eq / peak - 1).min())
    ret_total = float(eq[-1] / eq[0] - 1)
    final = float(INITIAL_CAPITAL * eq[-1] / eq[0])
    return {"sharpe": sharpe, "psr_0": psr_0, "max_dd": max_dd, "ret": ret_total, "final": final}


def main() -> None:
    if not PROBAS_CACHE.exists():
        raise SystemExit(f"cache ausente: {PROBAS_CACHE}. Rode exp_a1_threshold_search.py primeiro.")
    probas = pl.read_parquet(PROBAS_CACHE).to_pandas()
    probas["dt"] = pd.to_datetime(probas["dt"], utc=True)

    # MID-only: precisamos só de proba_mid válida (não exige proba_long)
    valid = probas["proba_mid"].notna()
    probas = probas[valid].reset_index(drop=True)
    print(f">>> probas válidas (MID): {len(probas):,} bars  "
          f"{probas['dt'].iloc[0]} → {probas['dt'].iloc[-1]}")

    dts_np = probas["dt"].values

    results = []
    for thr in THR_GRID:
        sim = simulate(probas, thr)
        val_m = metrics(sim["equity"], dts_np, "VAL")
        ho_m = metrics(sim["equity"], dts_np, "HOLDOUT")
        n_val = sum(1 for t in sim["trades"]
                    if (probas["dt"].iloc[t["entry_idx"]] >= TRAIN_START)
                    and (probas["dt"].iloc[t["entry_idx"]] <= VAL_END))
        n_ho = sum(1 for t in sim["trades"]
                   if probas["dt"].iloc[t["entry_idx"]] >= HOLDOUT_START)
        # meses no HO pra trades/mês
        ho_mask = dts_np >= np.datetime64(HOLDOUT_START)
        ho_days = (probas["dt"].iloc[-1] - probas.loc[ho_mask, "dt"].iloc[0]).total_seconds() / 86400 \
            if ho_mask.any() else 0
        ho_months = max(1e-6, ho_days / 30.0)
        results.append({
            "thr": thr,
            "val_sharpe": val_m["sharpe"], "val_psr0": val_m["psr_0"],
            "val_dd": val_m["max_dd"], "val_ret": val_m["ret"], "val_trades": n_val,
            "ho_sharpe": ho_m["sharpe"], "ho_psr0": ho_m["psr_0"],
            "ho_dd": ho_m["max_dd"], "ho_ret": ho_m["ret"], "ho_final": ho_m["final"],
            "ho_trades": n_ho, "ho_trades_per_month": n_ho / ho_months,
            "total_trades": sim["n_trades"],
        })

    df = pd.DataFrame(results)
    df_sorted = df.sort_values("val_sharpe", ascending=False).reset_index(drop=True)
    df_sorted.to_csv(OUT_CSV, index=False)
    print(f">>> resultados salvos em {OUT_CSV}\n")

    print("=" * 110)
    print(" SWEEP THRESHOLD — MID-only, no_bear=-0.05, COST=0.0015, FULL sizing")
    print(" Ranking POR VAL Sharpe (HOLDOUT só reportado, NAO escolhe)")
    print("=" * 110)
    show = df_sorted.copy()
    show["thr"] = show["thr"].map(lambda x: f"{x:.3f}")
    for c in ["val_sharpe", "ho_sharpe"]:
        show[c] = show[c].map(lambda x: f"{x:+.3f}" if not pd.isna(x) else "n/d")
    for c in ["val_dd", "ho_dd", "val_ret", "ho_ret"]:
        show[c] = show[c].map(lambda x: f"{100*x:+.1f}%" if not pd.isna(x) else "n/d")
    show["val_psr0"] = show["val_psr0"].map(lambda x: f"{x:.3f}" if not pd.isna(x) else "n/d")
    show["ho_psr0"] = show["ho_psr0"].map(lambda x: f"{x:.3f}" if not pd.isna(x) else "n/d")
    show["ho_final"] = show["ho_final"].map(lambda x: f"${x:,.0f}" if not pd.isna(x) else "n/d")
    show["ho_trades_per_month"] = show["ho_trades_per_month"].map(lambda x: f"{x:.1f}")
    cols = ["thr", "val_sharpe", "val_psr0", "val_dd", "val_ret", "val_trades",
            "ho_sharpe", "ho_psr0", "ho_dd", "ho_ret", "ho_trades", "ho_trades_per_month", "ho_final"]
    print(show[cols].to_string(index=False))

    # Winner: melhor VAL Sharpe com VAL trades >= 30
    winners = df_sorted[df_sorted["val_trades"] >= 30]
    print("\n" + "=" * 110)
    if winners.empty:
        print(" >>> NENHUM threshold passa constraint (VAL trades >= 30)")
    else:
        w = winners.iloc[0]
        cur = df_sorted[df_sorted["thr"] == 0.35].iloc[0]
        print(" WINNER (melhor VAL Sharpe com VAL trades >= 30):")
        print(f"   thr={w['thr']:.3f}")
        print(f"   VAL:     Sharpe {w['val_sharpe']:+.3f}  PSR0 {w['val_psr0']:.3f}  "
              f"DD {100*w['val_dd']:+.1f}%  trades {w['val_trades']:.0f}  ret {100*w['val_ret']:+.1f}%")
        print(f"   HOLDOUT: Sharpe {w['ho_sharpe']:+.3f}  PSR0 {w['ho_psr0']:.3f}  "
              f"DD {100*w['ho_dd']:+.1f}%  trades {w['ho_trades']:.0f} "
              f"({w['ho_trades_per_month']:.1f}/mes)  final ${w['ho_final']:,.0f}")
        print(f"\n CURRENT (prod, thr=0.35):")
        print(f"   VAL:     Sharpe {cur['val_sharpe']:+.3f}  trades {cur['val_trades']:.0f}")
        print(f"   HOLDOUT: Sharpe {cur['ho_sharpe']:+.3f}  trades {cur['ho_trades']:.0f} "
              f"({cur['ho_trades_per_month']:.1f}/mes)  final ${cur['ho_final']:,.0f}")
        # Veredito
        print("\n VEREDITO:")
        delta_val = w["val_sharpe"] - cur["val_sharpe"]
        delta_ho = w["ho_sharpe"] - cur["ho_sharpe"]
        delta_trades = w["ho_trades"] - cur["ho_trades"]
        if abs(w["thr"] - 0.35) < 1e-6:
            print("   thr=0.35 JA E o winner por VAL Sharpe -> MANTER thr=0.35")
        else:
            print(f"   thr={w['thr']:.3f} bate thr=0.35 no VAL (delta Sharpe {delta_val:+.3f})")
            print(f"   HOLDOUT: delta Sharpe {delta_ho:+.3f}  delta trades {delta_trades:+.0f}")
            if delta_ho >= -0.2 and w["ho_sharpe"] >= 1.0:
                print(f"   >>> RECOMENDAR TROCA: thr {w['thr']:.3f} oferece mais trades sem destruir HO Sharpe.")
            else:
                print(f"   >>> MANTER thr=0.35: winner do VAL nao se sustenta no HOLDOUT.")
    print("=" * 110)


if __name__ == "__main__":
    main()
