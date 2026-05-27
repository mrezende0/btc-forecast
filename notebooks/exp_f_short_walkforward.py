"""exp_f_short_walkforward — SHORT model honesto (walk-forward + uniqueness + HO split).

Pré-registrado no research_log §F.

Treina m_short com target (label==-1) em paralelo ao m_long (label==+1) atual.
Simula 3 estratégias: long-only, short-only, combined.

Espelha a estrutura do exp_a1_threshold_search.py (que gerou o cache walk_forward_probas.parquet)
mas adiciona o segundo modelo.

Cacheia data/walk_forward_probas_short.parquet pra reuso.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
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
THR_LONG = 0.35
THR_SHORT = 0.35
NO_BEAR = -0.05         # long: suprime se ret_30d < -5%
NO_BULL_SHORT = 0.05    # short: suprime se ret_30d > +5% (não shorta em bull)
BPD = 6
BARS_30D = 180
BARS_PER_YEAR = BPD * 365
RETRAIN_EVERY = 90 * BPD
START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)
VAL_END = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
HOLDOUT_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
INITIAL = 1000.0

LGB = dict(objective="binary", metric="binary_logloss", verbose=-1, n_jobs=-1,
           learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, lambda_l2=0.5)
N_ROUNDS = 500

CACHE_LONG = ROOT / "data" / "walk_forward_probas.parquet"  # já existe (do A1-A)
CACHE_SHORT = ROOT / "data" / "walk_forward_probas_short.parquet"


def build():
    df = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME, lag=1, asset="BTC").drop_nulls(subset=["atr_14"])
    lb = lab.triple_barrier(df, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=HORIZON)
    lb = lab.attach_uniqueness(lb, horizon_bars=HORIZON)
    lb = lb.with_columns(
        (pl.col("label") == 1).cast(pl.Int8).alias("y_long"),
        (pl.col("label") == -1).cast(pl.Int8).alias("y_short"),
    )
    excl = feat.LAG_SAFE_EXCLUDE | {"label", "hit_bar", "barrier_ret", "upper_px", "lower_px",
                                    "y_long", "y_short", "uniqueness_weight"}
    fcols = [c for c in lb.columns if c not in excl]
    base = ["open_time", "open", "high", "low", "close", "y_long", "y_short",
            "barrier_ret", "uniqueness_weight", "atr_14"]
    extra = [c for c in fcols if c not in base]
    m = lb.select(base + extra).drop_nulls(subset=fcols + ["y_long", "y_short"]).to_pandas()
    m["dt"] = pd.to_datetime(m["open_time"], unit="ms", utc=True)
    return m, fcols


def walk_forward_short(mat: pd.DataFrame, fcols: list[str]) -> np.ndarray:
    """Gera probas_short walk-forward expanding com retreino RETRAIN_EVERY bars."""
    n = len(mat)
    probas = np.full(n, np.nan)
    feat_arr = mat[fcols].to_numpy()
    y_arr = mat["y_short"].to_numpy()
    w_arr = mat["uniqueness_weight"].to_numpy()
    open_times = mat["open_time"].to_numpy()

    start_ms = int(START_DATE.timestamp() * 1000)
    start_pos = int(np.searchsorted(open_times, start_ms))

    model = None
    last_train_idx = -10**9
    t0 = time.time()
    for i in range(start_pos, n):
        if model is None or (i - last_train_idx) >= RETRAIN_EVERY:
            cut = i - HORIZON
            if cut > 500:
                X_tr = feat_arr[:cut]
                y_tr = y_arr[:cut]
                w_tr = w_arr[:cut]
                model = lgb.train(LGB, lgb.Dataset(X_tr, y_tr, weight=w_tr), num_boost_round=N_ROUNDS)
                last_train_idx = i
                print(f"  [short retrain] bar={i} dt={mat.iloc[i]['dt'].strftime('%Y-%m-%d')} ({time.time()-t0:.0f}s)")
        if model is not None:
            probas[i] = float(model.predict(feat_arr[i:i+1])[0])
    return probas


def ret_30d_series(closes: np.ndarray) -> np.ndarray:
    out = np.full(len(closes), np.nan)
    for i in range(BARS_30D, len(closes)):
        if closes[i - BARS_30D] > 0:
            out[i] = closes[i] / closes[i - BARS_30D] - 1
    return out


def simulate(mat: pd.DataFrame, proba_long: np.ndarray, proba_short: np.ndarray, mode: str) -> dict:
    """mode ∈ {long_only, short_only, combined}"""
    closes = mat["close"].to_numpy()
    highs = mat["high"].to_numpy()
    lows = mat["low"].to_numpy()
    atrs = mat["atr_14"].to_numpy()
    n = len(mat)
    r30 = ret_30d_series(closes)

    capital = INITIAL
    in_pos = False
    side = None
    entry_idx = -1
    entry_px = stop_px = target_px = np.nan
    expiry_idx = -1
    equity = np.full(n, INITIAL)
    trades = []

    for i in range(n):
        if in_pos:
            if side == "long":
                hit_target = highs[i] >= target_px
                hit_stop = lows[i] <= stop_px
            else:  # short
                hit_target = lows[i] <= target_px        # alvo short = abaixo
                hit_stop = highs[i] >= stop_px            # stop short = acima
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
                signed = px_ret if side == "long" else -px_ret
                net = signed - COST
                capital *= (1 + net)
                trades.append({"entry_idx": entry_idx, "exit_idx": i, "side": side, "net_ret": net})
                in_pos = False
                side = None

        if not in_pos and not (np.isnan(atrs[i])):
            sig_long = sig_short = False
            if mode in ("long_only", "combined") and not np.isnan(proba_long[i]):
                sig_long = proba_long[i] > THR_LONG
                if not np.isnan(r30[i]) and r30[i] < NO_BEAR:
                    sig_long = False
            if mode in ("short_only", "combined") and not np.isnan(proba_short[i]):
                sig_short = proba_short[i] > THR_SHORT
                if not np.isnan(r30[i]) and r30[i] > NO_BULL_SHORT:
                    sig_short = False
            # conflito: anula
            if sig_long and sig_short:
                sig_long = sig_short = False
            if sig_long:
                side = "long"
                entry_px = closes[i]
                target_px = entry_px + ATR_MULT * atrs[i]
                stop_px = entry_px - ATR_MULT * atrs[i]
                expiry_idx = i + HORIZON
                entry_idx = i
                in_pos = True
            elif sig_short:
                side = "short"
                entry_px = closes[i]
                target_px = entry_px - ATR_MULT * atrs[i]
                stop_px = entry_px + ATR_MULT * atrs[i]
                expiry_idx = i + HORIZON
                entry_idx = i
                in_pos = True

        if in_pos:
            px_ret = closes[i] / entry_px - 1
            signed = px_ret if side == "long" else -px_ret
            equity[i] = capital * (1 + signed - COST)
        else:
            equity[i] = capital

    return {"equity": equity, "trades": trades}


def seg_metrics(eq: np.ndarray, dts: np.ndarray, trades: list, seg: str) -> dict:
    if seg == "VAL":
        mask = (dts >= np.datetime64(START_DATE)) & (dts <= np.datetime64(VAL_END))
    elif seg == "HOLDOUT":
        mask = (dts >= np.datetime64(HOLDOUT_START))
    else:
        mask = np.ones(len(dts), dtype=bool)
    e = eq[mask]
    if len(e) < 30:
        return {"sharpe": np.nan, "max_dd": np.nan, "final": np.nan, "ret": np.nan, "n": 0, "win": 0}
    rets = np.diff(e) / e[:-1]
    rets = rets[~np.isnan(rets) & ~np.isinf(rets)]
    sharpe = float(rets.mean() / rets.std() * np.sqrt(BARS_PER_YEAR)) if rets.std() > 0 else 0.0
    peak = np.maximum.accumulate(e)
    max_dd = float((e / peak - 1).min())
    # trades dentro do segmento (compara em ms pra evitar timezone-aware vs naive)
    val_end_ms = np.datetime64(VAL_END.replace(tzinfo=None))
    ho_start_ms = np.datetime64(HOLDOUT_START.replace(tzinfo=None))
    start_ms = np.datetime64(START_DATE.replace(tzinfo=None))
    if seg == "VAL":
        seg_trades = [t for t in trades if start_ms <= dts[t["entry_idx"]] <= val_end_ms]
    elif seg == "HOLDOUT":
        seg_trades = [t for t in trades if dts[t["entry_idx"]] >= ho_start_ms]
    else:
        seg_trades = trades
    n = len(seg_trades)
    wins = sum(1 for t in seg_trades if t["net_ret"] > 0)
    return {"sharpe": sharpe, "max_dd": max_dd, "final": float(INITIAL * e[-1] / e[0]),
            "ret": float(e[-1] / e[0] - 1), "n": n, "win": (wins / n) if n else 0.0}


def main():
    # 1. Carrega cache LONG do A1-A
    if not CACHE_LONG.exists():
        raise SystemExit(f"cache long ausente: {CACHE_LONG}. Rode exp_a1_threshold_search.py antes.")
    long_cache = pl.read_parquet(CACHE_LONG).to_pandas()
    long_cache["dt"] = pd.to_datetime(long_cache["dt"], utc=True)

    # 2. Build matriz pro SHORT (mesma feature set, com y_short)
    print(">>> build matriz BTC 4h...", flush=True)
    t0 = time.time()
    mat, fcols = build()
    print(f"  rows={len(mat)} features={len(fcols)} ({time.time()-t0:.0f}s)")
    print(f"  base rates: long_win={100*mat['y_long'].mean():.1f}%  short_win={100*mat['y_short'].mean():.1f}%")

    # 3. Walk-forward SHORT (com cache)
    if CACHE_SHORT.exists():
        print(f">>> usando cache short {CACHE_SHORT}")
        sc = pl.read_parquet(CACHE_SHORT).to_pandas()
        # alinhar pelo open_time
        proba_short = mat["open_time"].map(dict(zip(sc["open_time"], sc["proba_short"]))).to_numpy()
    else:
        print(">>> walk-forward SHORT...", flush=True)
        proba_short = walk_forward_short(mat, fcols)
        pl.DataFrame({"open_time": mat["open_time"].to_numpy(), "proba_short": proba_short}).write_parquet(CACHE_SHORT)
        print(f">>> cache salvo em {CACHE_SHORT}")

    # 4. Alinhar proba_long do cache A1-A com mat (pode haver diff de bars warmup)
    proba_long_map = dict(zip(long_cache["open_time"], long_cache["proba_mid"]))
    proba_long = mat["open_time"].map(proba_long_map).to_numpy()
    proba_long = np.array([float(p) if p is not None and not pd.isna(p) else np.nan for p in proba_long])

    dts = mat["dt"].values

    # 5. Simula 3 modos
    print("\n" + "=" * 115)
    print(" EXP-F — SHORT model walk-forward honesto")
    print(f"  THR_LONG={THR_LONG}  THR_SHORT={THR_SHORT}  no_bear={NO_BEAR}  no_bull_short={NO_BULL_SHORT}")
    print("=" * 115)
    header = f"{'mode':<14}{'VAL Shp':>10}{'HO Shp':>10}{'VAL n':>8}{'HO n':>8}{'HO win%':>10}{'VAL DD':>10}{'HO DD':>10}{'HO final':>14}"
    print(header)
    print("-" * 115)

    results = {}
    for mode in ["long_only", "short_only", "combined"]:
        sim = simulate(mat, proba_long, proba_short, mode)
        val_m = seg_metrics(sim["equity"], dts, sim["trades"], "VAL")
        ho_m = seg_metrics(sim["equity"], dts, sim["trades"], "HOLDOUT")
        results[mode] = (val_m, ho_m, sim["trades"])
        row = (
            f"{mode:<14}"
            f"{val_m['sharpe']:>+10.3f}"
            f"{ho_m['sharpe']:>+10.3f}"
            f"{val_m['n']:>8d}"
            f"{ho_m['n']:>8d}"
            f"{100*ho_m['win']:>9.1f}%"
            f"{100*val_m['max_dd']:>+9.1f}%"
            f"{100*ho_m['max_dd']:>+9.1f}%"
            f"${ho_m['final']:>12,.0f}"
        )
        print(row)
    print("=" * 115)

    # 6. Decisão
    print("\nDECISÃO (critério pré-registrado):")
    ho_long = results["long_only"][1]["sharpe"]
    ho_short = results["short_only"][1]["sharpe"]
    ho_comb = results["combined"][1]["sharpe"]
    print(f"  HO Sharpe LONG-only:  {ho_long:+.2f}")
    print(f"  HO Sharpe SHORT-only: {ho_short:+.2f}")
    print(f"  HO Sharpe COMBINED:   {ho_comb:+.2f}")
    print(f"  Δ COMBINED vs LONG:   {ho_comb - ho_long:+.2f}")

    if ho_short >= 0.5 and ho_comb >= ho_long + 0.10:
        print("  >>> KEEP: SHORT passa critério (HO Sharpe ≥ 0.5 E combined > long + 0.10).")
        print("           Integrar em pipeline/model.py:predict_dual_horizon.")
    elif ho_short < 0.3:
        print("  >>> KILL: SHORT HO Sharpe < 0.3 — sem edge OOS honesto.")
    else:
        print("  >>> NEEDS-MORE-DATA: SHORT HO Sharpe ∈ [0.3, 0.5]. Não promove, monitora.")


if __name__ == "__main__":
    main()
