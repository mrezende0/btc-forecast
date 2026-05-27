"""exp_e2_ensemble_seeds — Ensemble de N LightGBMs com seeds diferentes.

Hipótese: LGB tem aleatoriedade (feature_fraction=0.8 + bagging_fraction=0.8 sem
seed fixo). Promediar probas de N modelos com seeds [42, 123, 456, 789, 999]
reduz variância da proba → Sharpe mais estável e potencialmente melhor.

Setup (MID-only em produção):
  thr=0.35, no_bear=-0.05, ENSEMBLE_RULE=MID, COST=0.0015, uniqueness weights, FULL sizing.
  Walk-forward: quarterly expanding (mesmo padrão exp_ensemble.py), purge=12 bars.

Compara:
  N=1 (baseline single seed=42)
  N=3 (média seeds [42, 123, 456])
  N=5 (média seeds [42, 123, 456, 789, 999])
  N=10 (5 originais + 5 novos seeds)
  + variância (std) entre os 5 single-seed runs no VAL/HO Sharpe.
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

# -------------------------------------------------------------- params (matching produção MID + exp_a1)
TIMEFRAME_MIN = 240
HORIZON_BARS = 12          # MID horizon (48h) — MID-only em prod
ATR_MULT = 3.0
COST = 0.0015
THRESHOLD = 0.35           # MID threshold validado
NO_BEAR = -0.05            # suprime sinal se BTC caiu >5% em 30d
INITIAL_CAPITAL = 1000.0
VAL_END = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
HOLDOUT_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
BARS_PER_DAY = 6
BARS_PER_YEAR = 6 * 365
BARS_30D = 180

BASE_LGB_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l2=0.5,
    verbose=-1,
    n_jobs=-1,
)
N_ROUNDS = 500

# Seeds: 5 primeiros = N=5; 5 extras para N=10.
SEEDS_BASE = [42, 123, 456, 789, 999]
SEEDS_EXTRA = [1337, 2718, 3141, 5772, 6022]
SEEDS_ALL = SEEDS_BASE + SEEDS_EXTRA  # 10 seeds


# -------------------------------------------------------------- matrix build
def build_matrix() -> tuple[pd.DataFrame, list[str]]:
    df = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME_MIN, lag=1).drop_nulls(subset=["atr_14"])
    labeled = lab.triple_barrier(df, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=HORIZON_BARS)
    labeled = lab.attach_uniqueness(labeled, horizon_bars=HORIZON_BARS)
    labeled = labeled.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y"))
    fc = [
        c for c in labeled.columns
        if c not in feat.LAG_SAFE_EXCLUDE
        and c not in {"label", "hit_bar", "barrier_ret", "upper_px", "lower_px", "y", "uniqueness_weight"}
    ]
    keep = ["open_time", "open", "high", "low", "close", "y", "barrier_ret", "uniqueness_weight", *fc]
    mat = labeled.select(keep).drop_nulls(subset=fc + ["y"]).to_pandas()
    mat["dt"] = pd.to_datetime(mat["open_time"], unit="ms", utc=True)
    mat["quarter"] = mat["dt"].dt.to_period("Q")
    return mat, fc


def make_params(seed: int) -> dict:
    p = dict(BASE_LGB_PARAMS)
    # Set all randomness sources to fixed seed for reproducibility per-seed.
    p["seed"] = seed
    p["feature_fraction_seed"] = seed
    p["bagging_seed"] = seed
    p["data_random_seed"] = seed
    p["extra_seed"] = seed
    p["deterministic"] = True
    return p


# -------------------------------------------------------------- walk-forward → probas per-seed
def walk_forward_probas(mat: pd.DataFrame, fc: list[str], seeds: list[int]) -> dict:
    """Quarterly expanding walk-forward; treina UM modelo por seed por trimestre.

    Retorna dict[seed] -> array(n_bars,) com proba (NaN onde não tem predição).
    Também retorna atr/high/low/close arrays para simulação.
    """
    quarters = [q for q in sorted(mat["quarter"].unique()) if q.start_time.year >= 2023]
    n = len(mat)
    fc_arr = mat[fc].to_numpy(dtype=np.float64)
    y_arr = mat["y"].to_numpy()
    w_arr = mat["uniqueness_weight"].to_numpy()

    probas_by_seed = {s: np.full(n, np.nan) for s in seeds}

    print(f">>> walk-forward expanding quarterly | seeds={seeds} | {len(quarters)} folds")
    t0 = time.time()
    for q in quarters:
        test_mask = mat["quarter"] == q
        test_idx = mat.index[test_mask].tolist()
        if not test_idx:
            continue
        test_start = test_idx[0]
        train_end = test_start - HORIZON_BARS
        test_use_start = test_start + HORIZON_BARS
        if train_end < 500 or test_use_start >= test_idx[-1]:
            continue
        train_idx = np.arange(0, train_end)
        test_use_idx = np.array([i for i in test_idx if i >= test_use_start])

        X_tr = fc_arr[train_idx]
        y_tr = y_arr[train_idx]
        w_tr = w_arr[train_idx]
        X_te = fc_arr[test_use_idx]

        t_q = time.time()
        for s in seeds:
            params = make_params(s)
            booster = lgb.train(
                params,
                lgb.Dataset(X_tr, y_tr, weight=w_tr),
                num_boost_round=N_ROUNDS,
            )
            probas_by_seed[s][test_use_idx] = booster.predict(X_te)
        elapsed = time.time() - t_q
        print(f"  {q!s:>8s}  tr={len(train_idx):>5d}  te={len(test_use_idx):>4d}  "
              f"{len(seeds)} seeds em {elapsed:.1f}s")

    print(f">>> walk-forward total: {time.time() - t0:.0f}s")
    return probas_by_seed


# -------------------------------------------------------------- simulação trades (mesma de exp_a1)
def simulate(mat: pd.DataFrame, proba: np.ndarray) -> dict:
    """Simula trades MID-only com thr=0.35, no_bear=-0.05, COST=0.0015, FULL sizing."""
    closes = mat["close"].to_numpy()
    highs = mat["high"].to_numpy()
    lows = mat["low"].to_numpy()
    atrs = mat["atr_14"].to_numpy()
    n = len(mat)

    # ret_30d (NO_BEAR)
    ret_30d = np.full(n, np.nan)
    for i in range(BARS_30D, n):
        if closes[i - BARS_30D] > 0:
            ret_30d[i] = closes[i] / closes[i - BARS_30D] - 1

    capital = INITIAL_CAPITAL
    in_pos = False
    entry_idx = -1
    entry_px = stop_px = target_px = np.nan
    expiry_idx = -1
    equity = np.full(n, INITIAL_CAPITAL)
    trades = []

    for i in range(n):
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

        if not in_pos and not (np.isnan(proba[i]) or np.isnan(atrs[i])):
            sig = proba[i] > THRESHOLD
            if not np.isnan(ret_30d[i]) and ret_30d[i] < NO_BEAR:
                sig = False
            if sig:
                entry_idx = i
                entry_px = closes[i]
                target_px = entry_px + ATR_MULT * atrs[i]
                stop_px = entry_px - ATR_MULT * atrs[i]
                expiry_idx = i + HORIZON_BARS
                in_pos = True

        if in_pos:
            unreal = closes[i] / entry_px - 1
            equity[i] = capital * (1 + unreal - COST)
        else:
            equity[i] = capital

    if in_pos:
        net = closes[-1] / entry_px - 1 - COST
        capital *= (1 + net)
        trades.append({"entry_idx": entry_idx, "exit_idx": n - 1, "net_ret": net})
        equity[-1] = capital

    return {"equity": equity, "n_trades": len(trades), "trades": trades, "final_capital": float(capital)}


def metrics(equity: np.ndarray, mat: pd.DataFrame, segment: str) -> dict:
    dts = mat["dt"].values
    if segment == "VAL":
        mask = (dts <= np.datetime64(VAL_END))
    elif segment == "HOLDOUT":
        mask = (dts >= np.datetime64(HOLDOUT_START))
    else:
        mask = np.ones(len(dts), dtype=bool)
    seg = equity[mask]
    if len(seg) < 30:
        return {"sharpe": np.nan, "max_dd": np.nan, "n_bars": len(seg), "ret_total": np.nan, "final": np.nan}
    rets = np.diff(seg) / seg[:-1]
    rets = rets[~np.isnan(rets)]
    if len(rets) < 30 or rets.std() == 0:
        return {"sharpe": 0.0, "max_dd": 0.0, "n_bars": len(seg),
                "ret_total": float(seg[-1] / seg[0] - 1), "final": float(seg[-1])}
    sharpe = float(rets.mean() / rets.std() * np.sqrt(BARS_PER_YEAR))
    peak = np.maximum.accumulate(seg)
    max_dd = float((seg / peak - 1).min())
    return {
        "sharpe": sharpe,
        "max_dd": max_dd,
        "n_bars": len(seg),
        "ret_total": float(seg[-1] / seg[0] - 1),
        "final": float(seg[-1]),
    }


def count_trades_by_segment(trades, mat) -> tuple[int, int]:
    dts = mat["dt"].values
    n_val = 0
    n_ho = 0
    for t in trades:
        d = dts[t["entry_idx"]]
        if d <= np.datetime64(VAL_END):
            n_val += 1
        elif d >= np.datetime64(HOLDOUT_START):
            n_ho += 1
    return n_val, n_ho


# -------------------------------------------------------------- runner
def evaluate_config(name: str, proba: np.ndarray, mat: pd.DataFrame) -> dict:
    sim = simulate(mat, proba)
    val_m = metrics(sim["equity"], mat, "VAL")
    ho_m = metrics(sim["equity"], mat, "HOLDOUT")
    full_m = metrics(sim["equity"], mat, "FULL")
    n_val, n_ho = count_trades_by_segment(sim["trades"], mat)
    return {
        "config": name,
        "val_sharpe": val_m["sharpe"],
        "ho_sharpe": ho_m["sharpe"],
        "n_trades_total": sim["n_trades"],
        "n_trades_val": n_val,
        "n_trades_ho": n_ho,
        "final_dollar": sim["final_capital"],
        "val_dd": val_m["max_dd"],
        "ho_dd": ho_m["max_dd"],
        "full_ret": full_m["ret_total"],
    }


def main():
    t_start = time.time()
    print(">>> construindo matriz (h=12, MID-only) ...")
    mat, fc = build_matrix()
    print(f"  matriz: {len(mat):,} rows  features={len(fc)}")

    # Roda walk-forward para TODOS os 10 seeds (uma só passada).
    probas = walk_forward_probas(mat, fc, SEEDS_ALL)

    # Ensemble: média das probas dos seeds selecionados
    def ensemble_proba(seeds: list[int]) -> np.ndarray:
        arr = np.stack([probas[s] for s in seeds], axis=0)
        # nanmean ignora NaN (mas todas as colunas deveriam ter mesmo padrão de NaN)
        return np.nanmean(arr, axis=0)

    configs = [
        ("N=1 (seed=42 baseline)", probas[42]),
        ("N=3 (seeds[42,123,456])", ensemble_proba(SEEDS_BASE[:3])),
        ("N=5 (seeds base)", ensemble_proba(SEEDS_BASE)),
        ("N=10 (seeds base+extra)", ensemble_proba(SEEDS_ALL)),
    ]

    print("\n>>> simulando configs ensemble ...")
    results = [evaluate_config(name, p, mat) for name, p in configs]

    # Variance entre seeds individuais (5 singles base)
    print("\n>>> simulando cada single-seed (variance entre seeds) ...")
    single_results = []
    for s in SEEDS_BASE:
        r = evaluate_config(f"single seed={s}", probas[s], mat)
        single_results.append(r)
        print(f"  seed={s:>4d}  VAL Sharpe {r['val_sharpe']:+.3f}  HO Sharpe {r['ho_sharpe']:+.3f}  "
              f"final ${r['final_dollar']:.0f}  trades={r['n_trades_total']}")

    val_sharpes = np.array([r["val_sharpe"] for r in single_results])
    ho_sharpes = np.array([r["ho_sharpe"] for r in single_results])
    finals = np.array([r["final_dollar"] for r in single_results])
    var_summary = {
        "val_sharpe_mean": float(val_sharpes.mean()),
        "val_sharpe_std": float(val_sharpes.std(ddof=1)),
        "ho_sharpe_mean": float(ho_sharpes.mean()),
        "ho_sharpe_std": float(ho_sharpes.std(ddof=1)),
        "final_mean": float(finals.mean()),
        "final_std": float(finals.std(ddof=1)),
    }

    # ---- print tabelas ----
    print("\n" + "=" * 110)
    print(" RESULTADOS — Ensemble de seeds (MID-only, thr=0.35, no_bear=-0.05, COST=0.0015, FULL sizing)")
    print("=" * 110)
    hdr = f"{'config':<28s} {'VAL Sh':>8s} {'HO Sh':>8s} {'trades':>7s} {'tr VAL':>7s} {'tr HO':>7s} {'final $1k':>10s} {'HO DD':>8s}"
    print(hdr)
    print("-" * 110)
    for r in results:
        print(f"{r['config']:<28s} {r['val_sharpe']:>+8.3f} {r['ho_sharpe']:>+8.3f} "
              f"{r['n_trades_total']:>7d} {r['n_trades_val']:>7d} {r['n_trades_ho']:>7d} "
              f"{r['final_dollar']:>10.0f} {100*r['ho_dd']:>+7.1f}%")

    print("\n" + "=" * 110)
    print(" VARIÂNCIA entre 5 single-seed runs (seeds = 42, 123, 456, 789, 999)")
    print("=" * 110)
    print(f"  VAL Sharpe  mean = {var_summary['val_sharpe_mean']:+.3f}   std = {var_summary['val_sharpe_std']:.3f}   "
          f"range = [{val_sharpes.min():+.3f}, {val_sharpes.max():+.3f}]")
    print(f"  HO  Sharpe  mean = {var_summary['ho_sharpe_mean']:+.3f}   std = {var_summary['ho_sharpe_std']:.3f}   "
          f"range = [{ho_sharpes.min():+.3f}, {ho_sharpes.max():+.3f}]")
    print(f"  Final $1k   mean = {var_summary['final_mean']:.0f}      std = {var_summary['final_std']:.0f}      "
          f"range = [{finals.min():.0f}, {finals.max():.0f}]")

    # ---- save CSV ----
    out_csv = ROOT / "data" / "exp_e2_ensemble_seeds_results.csv"
    df_out = pd.DataFrame(results + single_results)
    df_out.to_csv(out_csv, index=False)
    print(f"\n>>> CSV salvo em {out_csv}")
    print(f">>> tempo total: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
