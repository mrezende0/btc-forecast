"""exp_e_dynamic_leverage — Backtest contrafactual de alavancagem dinâmica.

Aplica `mdl.dynamic_leverage(proba_mid, rv_30d_ann)` por trade em cima das probas
JÁ CACHEADAS em data/walk_forward_probas.parquet (winner A1-A).

Sem K extra: não escolhe params, só aplica a fórmula sobre dado existente.

Compara 4 modos:
  - flat_1x       : leverage=1 (baseline atual em prod sem env override)
  - flat_3x       : leverage=3 fixo (override usuário típico agressivo)
  - dynamic       : leverage(proba, vol) ∈ [1, 5]
  - dynamic_cap3  : leverage(proba, vol) clamped em [1, 3] (conservador)

Métricas: VAL Sharpe / HOLDOUT Sharpe / final $1k / MaxDD / risk per trade médio.
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

from pipeline import model as mdl  # noqa: E402

HORIZON_MID = 12
ATR_MULT = 3.0
COST = 0.0015
THR = 0.35
NO_BEAR = -0.05
INITIAL = 1000.0
BARS_PER_YEAR = 6 * 365
BARS_30D = 180

VAL_END = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
HOLDOUT_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
TRAIN_START = datetime(2023, 1, 1, tzinfo=timezone.utc)

PROBAS_CACHE = ROOT / "data" / "walk_forward_probas.parquet"


def simulate(probas: pd.DataFrame, lev_mode: str) -> dict:
    closes = probas["close"].to_numpy()
    highs = probas["high"].to_numpy()
    lows = probas["low"].to_numpy()
    atrs = probas["atr"].to_numpy()
    pm = probas["proba_mid"].to_numpy()
    ret_30d = probas["ret_30d"].to_numpy()
    n = len(probas)

    # Vol anualizada rolling 30d → input pro dynamic_leverage
    log_ret = np.diff(np.log(closes), prepend=np.log(closes[0]))
    rv_series = pd.Series(log_ret).rolling(BARS_30D).std().to_numpy()
    rv_ann = rv_series * np.sqrt(BARS_PER_YEAR)

    capital = INITIAL
    in_pos = False
    entry_idx = -1
    entry_px = stop_px = target_px = np.nan
    expiry_idx = -1
    leverage_used = 1.0
    equity = np.full(n, INITIAL)
    trades = []

    for i in range(n):
        if in_pos:
            hit_target = highs[i] >= target_px
            hit_stop = lows[i] <= stop_px
            timed_out = i >= expiry_idx
            exit_now = False
            exit_px = np.nan
            if hit_stop:
                exit_px = stop_px; exit_now = True
            elif hit_target:
                exit_px = target_px; exit_now = True
            elif timed_out:
                exit_px = closes[i]; exit_now = True
            if exit_now:
                px_ret = exit_px / entry_px - 1
                net = leverage_used * px_ret - COST  # leverage amplifica retorno bruto, custo round-trip fixo
                capital *= (1 + net)
                trades.append({"entry_idx": entry_idx, "exit_idx": i, "net_ret": net,
                               "leverage": leverage_used, "px_ret": px_ret})
                in_pos = False

        if not in_pos and not (np.isnan(pm[i]) or np.isnan(atrs[i])):
            sig = pm[i] > THR
            in_bear = (not np.isnan(ret_30d[i])) and (ret_30d[i] < NO_BEAR)
            if sig and not in_bear:
                # Calcula leverage
                if lev_mode == "flat_1x":
                    lev = 1.0
                elif lev_mode == "flat_3x":
                    lev = 3.0
                elif lev_mode == "dynamic":
                    info = mdl.dynamic_leverage(pm[i], rv_ann[i] if not np.isnan(rv_ann[i]) else None,
                                                in_bear=False, leverage_max=5.0)
                    lev = info["leverage"]
                elif lev_mode == "dynamic_cap3":
                    info = mdl.dynamic_leverage(pm[i], rv_ann[i] if not np.isnan(rv_ann[i]) else None,
                                                in_bear=False, leverage_max=3.0)
                    lev = info["leverage"]
                else:
                    raise ValueError(lev_mode)
                entry_idx = i
                entry_px = closes[i]
                target_px = entry_px + ATR_MULT * atrs[i]
                stop_px = entry_px - ATR_MULT * atrs[i]
                expiry_idx = i + HORIZON_MID
                leverage_used = lev
                in_pos = True

        if in_pos:
            unreal = closes[i] / entry_px - 1
            equity[i] = capital * (1 + leverage_used * unreal - COST)
        else:
            equity[i] = capital

    return {"equity": equity, "trades": trades, "rv_ann": rv_ann}


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

    modes = ["flat_1x", "flat_3x", "dynamic", "dynamic_cap3"]
    print("=" * 115)
    print(" EXP-E — Dynamic Leverage Backtest (cache A1-A, sem K extra)")
    print(f"  thr={THR}  no_bear={NO_BEAR}  cost={COST}  horizon={HORIZON_MID}bars  vol_target={mdl.LEVERAGE_VOL_TARGET}")
    print("=" * 115)
    header = f"{'mode':<16}{'VAL Shp':>10}{'HO Shp':>10}{'VAL DD':>10}{'HO DD':>10}{'VAL final':>14}{'HO final':>14}{'avg lev':>10}{'n trades':>10}"
    print(header)
    print("-" * 115)

    results = []
    for m in modes:
        sim = simulate(probas, m)
        val_m = seg_metrics(sim["equity"], dts, "VAL")
        ho_m = seg_metrics(sim["equity"], dts, "HOLDOUT")
        levs = [t["leverage"] for t in sim["trades"]]
        avg_lev = float(np.mean(levs)) if levs else 0.0
        row = (
            f"{m:<16}"
            f"{val_m['sharpe']:>+10.3f}"
            f"{ho_m['sharpe']:>+10.3f}"
            f"{100*val_m['max_dd']:>+9.1f}%"
            f"{100*ho_m['max_dd']:>+9.1f}%"
            f"${val_m['final']:>12,.0f}"
            f"${ho_m['final']:>12,.0f}"
            f"{avg_lev:>10.2f}"
            f"{len(sim['trades']):>10d}"
        )
        print(row)
        results.append({"mode": m, "val_sharpe": val_m["sharpe"], "ho_sharpe": ho_m["sharpe"],
                        "val_dd": val_m["max_dd"], "ho_dd": ho_m["max_dd"],
                        "val_final": val_m["final"], "ho_final": ho_m["final"],
                        "avg_lev": avg_lev, "n_trades": len(sim["trades"])})
    print("=" * 115)

    # Decisão
    base = next(r for r in results if r["mode"] == "flat_1x")
    dyn = next(r for r in results if r["mode"] == "dynamic")
    cap3 = next(r for r in results if r["mode"] == "dynamic_cap3")
    flat3 = next(r for r in results if r["mode"] == "flat_3x")

    print("\nDECISÃO:")
    print(f"  Baseline flat_1x:  HO Sharpe {base['ho_sharpe']:+.2f}  final ${base['ho_final']:,.0f}  DD {100*base['ho_dd']:+.1f}%")
    print(f"  dynamic [1, 5]:    HO Sharpe {dyn['ho_sharpe']:+.2f}  final ${dyn['ho_final']:,.0f}  DD {100*dyn['ho_dd']:+.1f}%  (avg lev {dyn['avg_lev']:.2f})")
    print(f"  dynamic [1, 3]:    HO Sharpe {cap3['ho_sharpe']:+.2f}  final ${cap3['ho_final']:,.0f}  DD {100*cap3['ho_dd']:+.1f}%  (avg lev {cap3['avg_lev']:.2f})")
    print(f"  flat_3x:           HO Sharpe {flat3['ho_sharpe']:+.2f}  final ${flat3['ho_final']:,.0f}  DD {100*flat3['ho_dd']:+.1f}%")

    # Comparação leverage dinâmico vs flat na mesma escala (avg_lev)
    print(f"\n  dynamic vs flat na mesma escala (avg lev ~{dyn['avg_lev']:.1f}):")
    if dyn["ho_sharpe"] >= base["ho_sharpe"] - 0.1 and dyn["ho_final"] > base["ho_final"]:
        print(f"  >>> dynamic OK: mantém Sharpe vs baseline e amplia retorno via leverage condicional")
    else:
        print(f"  >>> dynamic NEEDS REVIEW: Sharpe ou retorno pior que baseline")


if __name__ == "__main__":
    main()
