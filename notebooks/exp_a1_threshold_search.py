"""exp_a1_threshold_search — Caminho A do A1.

Re-escolhe threshold + NO_BEAR + ensemble rule HONESTAMENTE no VAL apenas,
congela, reporta HOLDOUT sem retoque.

Pipeline:
1. Walk-forward 1x igual exp_backtest_1k (uniqueness + COST=0.0015), mas
   sempre prediz proba_mid e proba_long pra TODA barra (não só quando flat).
   Salva data/walk_forward_probas.parquet (cache — pula se já existe).
2. Grid: thr_mid × thr_long × no_bear × rule (200 configs).
3. Pra cada config: simula trades, computa Sharpe + PSR(0) no VAL + HOLDOUT.
4. Ranking POR VAL APENAS (HOLDOUT só reportado, NÃO escolhe).
5. Winner = melhor VAL Sharpe com VAL Sharpe ≥ 0.3 e n_trades_val ≥ 30.

K incremental: +1 (tuning A é 1 entrada, não 200 — pre-registrado).
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

# ----------------------------------------------------------------- params (mesmo exp_backtest_1k)
TIMEFRAME_MIN = 240
HORIZON_MID = 12
HORIZON_LONG = 18
ATR_MULT = 3.0
COST = 0.0015
BARS_PER_DAY = 6
RETRAIN_EVERY_BARS = 90 * BARS_PER_DAY
START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)
VAL_END = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
HOLDOUT_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
INITIAL_CAPITAL = 1000.0

LGB_PARAMS = dict(
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
BARS_30D = 180  # 30 dias × 6 bars 4h

PROBAS_CACHE = ROOT / "data" / "walk_forward_probas.parquet"


# ----------------------------------------------------------------- step 1: probas
def build_matrix(horizon_bars: int) -> tuple[pd.DataFrame, list[str]]:
    df = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME_MIN, lag=1).drop_nulls(subset=["atr_14"])
    labeled = lab.triple_barrier(df, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=horizon_bars)
    labeled = lab.attach_uniqueness(labeled, horizon_bars=horizon_bars)
    labeled = labeled.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y"))
    fc = [
        c for c in labeled.columns
        if c not in feat.LAG_SAFE_EXCLUDE
        and c not in {"label", "hit_bar", "barrier_ret", "upper_px", "lower_px", "y", "uniqueness_weight"}
    ]
    keep = ["open_time", "open", "high", "low", "close", "y", "uniqueness_weight", *fc]
    mat = labeled.select(keep).drop_nulls(subset=fc + ["y"]).to_pandas()
    return mat, fc


def generate_walk_forward_probas() -> pd.DataFrame:
    """Reproduz walk-forward de exp_backtest_1k mas SEMPRE prediz proba (não só flat)."""
    print(">>> construindo matrizes (h=12, h=18) ...")
    t0 = time.time()
    mat_mid, fc_mid = build_matrix(HORIZON_MID)
    mat_long, fc_long = build_matrix(HORIZON_LONG)
    print(f"  mid: {len(mat_mid):,} rows  long: {len(mat_long):,} rows  ({time.time() - t0:.1f}s)")

    mat_mid["dt"] = pd.to_datetime(mat_mid["open_time"], unit="ms", utc=True)
    mat_long_idx = mat_long.set_index("open_time")
    fc_mid_arr = mat_mid[fc_mid].to_numpy()
    fc_long_df = mat_long_idx[fc_long]
    open_times = mat_mid["open_time"].to_numpy()
    closes = mat_mid["close"].to_numpy()
    highs = mat_mid["high"].to_numpy()
    lows = mat_mid["low"].to_numpy()
    atrs = mat_mid["atr_14"].to_numpy()
    y_mid_arr = mat_mid["y"].to_numpy()
    w_mid_arr = mat_mid["uniqueness_weight"].to_numpy()

    start_ms = int(START_DATE.timestamp() * 1000)
    start_pos = mat_mid["open_time"].searchsorted(start_ms)
    n_bars = len(mat_mid)
    print(f">>> simulação de probas inicia em pos={start_pos} dt={mat_mid.iloc[start_pos]['dt']}")

    model_mid: lgb.Booster | None = None
    model_long: lgb.Booster | None = None
    last_train_idx = -10**9

    probas_mid = np.full(n_bars, np.nan)
    probas_long = np.full(n_bars, np.nan)

    t_loop = time.time()
    for i in range(start_pos, n_bars):
        if model_mid is None or (i - last_train_idx) >= RETRAIN_EVERY_BARS:
            cut_mid = i - HORIZON_MID
            if cut_mid > 500:
                X_tr = fc_mid_arr[:cut_mid]
                y_tr = y_mid_arr[:cut_mid]
                w_tr = w_mid_arr[:cut_mid]
                model_mid = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, y_tr, weight=w_tr), num_boost_round=N_ROUNDS)
                cutoff_ot = open_times[i - HORIZON_LONG] if i - HORIZON_LONG >= 0 else open_times[0]
                mask_long = mat_long["open_time"] < cutoff_ot
                X_trl = mat_long.loc[mask_long, fc_long].to_numpy()
                y_trl = mat_long.loc[mask_long, "y"].to_numpy()
                w_trl = mat_long.loc[mask_long, "uniqueness_weight"].to_numpy()
                if len(X_trl) > 500:
                    model_long = lgb.train(LGB_PARAMS, lgb.Dataset(X_trl, y_trl, weight=w_trl), num_boost_round=N_ROUNDS)
                    last_train_idx = i
                    elapsed = time.time() - t_loop
                    print(f"  [retreino] bar={i} dt={mat_mid.iloc[i]['dt'].strftime('%Y-%m-%d')} ({elapsed:.0f}s)")

        if model_mid is not None and model_long is not None:
            probas_mid[i] = float(model_mid.predict(fc_mid_arr[i : i + 1])[0])
            ot_i = open_times[i]
            if ot_i in fc_long_df.index:
                probas_long[i] = float(model_long.predict(fc_long_df.loc[[ot_i]].to_numpy())[0])
            else:
                probas_long[i] = np.nan

    # ret_30d pra NO_BEAR
    ret_30d = np.full(n_bars, np.nan)
    for i in range(BARS_30D, n_bars):
        if closes[i - BARS_30D] > 0:
            ret_30d[i] = closes[i] / closes[i - BARS_30D] - 1

    out = pd.DataFrame({
        "open_time": open_times,
        "dt": mat_mid["dt"].values,
        "open": mat_mid["open"].values,
        "high": highs,
        "low": lows,
        "close": closes,
        "atr": atrs,
        "proba_mid": probas_mid,
        "proba_long": probas_long,
        "ret_30d": ret_30d,
    })
    PROBAS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(out).write_parquet(PROBAS_CACHE)
    print(f">>> probas salvas em {PROBAS_CACHE} ({len(out):,} rows)")
    return out


# ----------------------------------------------------------------- step 2: simulate (vetorizado pra grid)
def simulate(probas: pd.DataFrame, thr_mid: float, thr_long: float,
             no_bear: float | None, rule: str) -> dict:
    """Simula trade walk-forward dado um config. Retorna equity series + métricas."""
    closes = probas["close"].to_numpy()
    highs = probas["high"].to_numpy()
    lows = probas["low"].to_numpy()
    atrs = probas["atr"].to_numpy()
    pm = probas["proba_mid"].to_numpy()
    pl_arr = probas["proba_long"].to_numpy()
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
        # exit logic
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

        # entry logic
        if not in_pos and not (np.isnan(pm[i]) or np.isnan(pl_arr[i]) or np.isnan(atrs[i])):
            sig_mid = pm[i] > thr_mid
            sig_long = pl_arr[i] > thr_long
            if rule == "AND":
                sig = sig_mid and sig_long
            elif rule == "OR":
                sig = sig_mid or sig_long
            elif rule == "MID":
                sig = sig_mid
            elif rule == "LONG":
                sig = sig_long
            else:
                sig = False
            if no_bear is not None and not np.isnan(ret_30d[i]):
                if ret_30d[i] < no_bear:
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

    # close final
    if in_pos:
        net = closes[-1] / entry_px - 1 - COST
        capital *= (1 + net)
        trades.append({"entry_idx": entry_idx, "exit_idx": n - 1, "net_ret": net})
        equity[-1] = capital
    return {
        "equity": equity,
        "n_trades": len(trades),
        "trades": trades,
    }


def metrics(equity: np.ndarray, probas: pd.DataFrame, segment: str) -> dict:
    """Sharpe + PSR(0) por segmento."""
    bars_per_year = 6 * 365
    dts = probas["dt"].values
    if segment == "VAL":
        mask = (dts <= np.datetime64(VAL_END))
    elif segment == "HOLDOUT":
        mask = (dts >= np.datetime64(HOLDOUT_START))
    else:
        mask = np.ones(len(dts), dtype=bool)
    seg = equity[mask]
    if len(seg) < 30:
        return {"sharpe": np.nan, "psr_0": np.nan, "max_dd": np.nan, "n_bars": len(seg), "ret_total": np.nan}
    rets = np.diff(seg) / seg[:-1]
    rets = rets[~np.isnan(rets)]
    if len(rets) < 30 or rets.std() == 0:
        return {"sharpe": 0.0, "psr_0": 0.5, "max_dd": 0.0, "n_bars": len(seg), "ret_total": float(seg[-1] / seg[0] - 1)}
    sharpe = float(rets.mean() / rets.std() * np.sqrt(bars_per_year))
    # PSR(0) Bailey-LdP
    from scipy.stats import norm
    sr_per_bar = rets.mean() / rets.std()
    skew = float(pd.Series(rets).skew())
    kurt = float(pd.Series(rets).kurtosis())
    denom = np.sqrt(max(1e-12, (1 - skew * sr_per_bar + (kurt) / 4 * sr_per_bar ** 2) / (len(rets) - 1)))
    z = sr_per_bar / denom
    psr_0 = float(norm.cdf(z))
    peak = np.maximum.accumulate(seg)
    max_dd = float((seg / peak - 1).min())
    ret_total = float(seg[-1] / seg[0] - 1)
    return {"sharpe": sharpe, "psr_0": psr_0, "max_dd": max_dd, "n_bars": len(seg), "ret_total": ret_total}


# ----------------------------------------------------------------- step 3: grid
def main() -> None:
    if PROBAS_CACHE.exists():
        print(f">>> usando cache {PROBAS_CACHE}")
        probas = pl.read_parquet(PROBAS_CACHE).to_pandas()
        probas["dt"] = pd.to_datetime(probas["dt"], utc=True)
    else:
        probas = generate_walk_forward_probas()
        probas["dt"] = pd.to_datetime(probas["dt"], utc=True)

    # filtra só bars com proba válida
    valid_mask = probas["proba_mid"].notna() & probas["proba_long"].notna()
    probas = probas[valid_mask].reset_index(drop=True)
    print(f">>> probas válidas: {len(probas):,} bars  período {probas['dt'].iloc[0]} → {probas['dt'].iloc[-1]}")

    grid = []
    for thr_mid in [0.30, 0.35, 0.40, 0.45, 0.50]:
        for thr_long in [0.30, 0.35, 0.40, 0.45, 0.50]:
            for no_bear in [None, -0.10, -0.05, 0.00]:
                for rule in ["AND", "OR", "MID", "LONG"]:
                    grid.append((thr_mid, thr_long, no_bear, rule))

    print(f">>> grid: {len(grid)} configs")
    results = []
    t0 = time.time()
    for j, (thr_m, thr_l, nb, rule) in enumerate(grid):
        sim = simulate(probas, thr_m, thr_l, nb, rule)
        val_m = metrics(sim["equity"], probas, "VAL")
        ho_m = metrics(sim["equity"], probas, "HOLDOUT")
        # n_trades por segmento (aproxima)
        n_trades_val = sum(1 for t in sim["trades"] if probas["dt"].iloc[t["entry_idx"]] <= VAL_END)
        n_trades_ho = sum(1 for t in sim["trades"] if probas["dt"].iloc[t["entry_idx"]] >= HOLDOUT_START)
        results.append({
            "thr_mid": thr_m, "thr_long": thr_l, "no_bear": nb if nb is not None else "off", "rule": rule,
            "val_sharpe": val_m["sharpe"], "val_psr0": val_m["psr_0"], "val_dd": val_m["max_dd"],
            "val_ret": val_m["ret_total"], "n_trades_val": n_trades_val,
            "ho_sharpe": ho_m["sharpe"], "ho_psr0": ho_m["psr_0"], "ho_dd": ho_m["max_dd"],
            "ho_ret": ho_m["ret_total"], "n_trades_ho": n_trades_ho,
            "n_trades_total": sim["n_trades"],
        })
        if (j + 1) % 50 == 0:
            print(f"  {j + 1}/{len(grid)} configs ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(results)
    df_sorted = df.sort_values("val_sharpe", ascending=False).reset_index(drop=True)
    out_csv = ROOT / "data" / "a1_threshold_search_results.csv"
    df_sorted.to_csv(out_csv, index=False)
    print(f"\n>>> resultados salvos em {out_csv}")

    print("\n" + "=" * 95)
    print(" TOP-10 por VAL Sharpe (ranking honesto — HOLDOUT só pra reportar, NÃO escolhe)")
    print("=" * 95)
    cols = ["thr_mid", "thr_long", "no_bear", "rule", "val_sharpe", "val_psr0", "val_dd", "n_trades_val",
            "ho_sharpe", "ho_psr0", "ho_dd", "n_trades_ho"]
    top10 = df_sorted.head(10)[cols].copy()
    for c in ["val_sharpe", "val_psr0", "val_dd", "ho_sharpe", "ho_psr0", "ho_dd"]:
        top10[c] = top10[c].map(lambda x: f"{x:+.2f}" if not pd.isna(x) else "n/d")
    print(top10.to_string(index=False))

    # winner constraint
    winners = df_sorted[(df_sorted["val_sharpe"] >= 0.3) & (df_sorted["n_trades_val"] >= 30)]
    print("\n" + "=" * 95)
    if winners.empty:
        print(" >>> NENHUM CONFIG passa constraints (VAL Sharpe ≥ 0.3 + n_trades_val ≥ 30)")
        print(" >>> CAMINHO A FALHA. Recomendação ROADMAP: Caminho B (arquivar) ou Caminho C (apostar em features).")
    else:
        w = winners.iloc[0]
        print(f" >>> WINNER (melhor VAL Sharpe com VAL ≥ 0.3 e n_trades_val ≥ 30):")
        print(f"     thr_mid={w['thr_mid']:.2f}  thr_long={w['thr_long']:.2f}  no_bear={w['no_bear']}  rule={w['rule']}")
        print(f"     VAL:     Sharpe {w['val_sharpe']:+.2f}  PSR(0) {w['val_psr0']:.3f}  MaxDD {100*w['val_dd']:.1f}%  trades {w['n_trades_val']:.0f}  ret {100*w['val_ret']:+.1f}%")
        print(f"     HOLDOUT: Sharpe {w['ho_sharpe']:+.2f}  PSR(0) {w['ho_psr0']:.3f}  MaxDD {100*w['ho_dd']:.1f}%  trades {w['n_trades_ho']:.0f}  ret {100*w['ho_ret']:+.1f}%")
        if w["ho_sharpe"] >= 0.7 * max(0.01, w["val_sharpe"]):
            print(f"     >>> HOLDOUT robusto (≥ 70% do VAL) — sinal real candidato. Próximo: Caminho C (A2 features de fluxo).")
        elif w["ho_sharpe"] >= 0.3:
            print(f"     >>> HOLDOUT degrada vs VAL mas positivo. Cauteloso — pode ser overfit moderado.")
        else:
            print(f"     >>> HOLDOUT degrada drasticamente → VAL provavelmente overfit. Caminho B mais honesto.")

    print("=" * 95)


if __name__ == "__main__":
    main()
