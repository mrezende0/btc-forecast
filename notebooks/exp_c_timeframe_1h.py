"""exp_c_timeframe_1h — Experimento: timeframe 1h vs 4h baseline.

Hipótese: 1h gera ~4x mais sinais que 4h sem o ruído extremo de 15m (Sharpe -24).
Macro (DXY/VIX) é daily e ainda alinha bem em grid 1h.

Setup identico ao baseline 4h, ajustado pro novo timeframe:
  - TIMEFRAME_MIN = 60
  - HORIZON_MID = 48 bars (48h, mesmo horizonte temporal que 4h × 12)
  - HORIZON_LONG = 72 bars (72h, mesmo horizonte que 4h × 18)
  - BARS_PER_DAY = 24
  - BARS_30D = 720
  - RETRAIN_EVERY_BARS = 90 × 24 = 2160
  - Mesmo LGB params, mesmo COST=0.0015, ATR_MULT=3.0
  - Mesmo MID-only rule, threshold 0.35, no_bear=-0.05
  - Mesma estrutura walk-forward expanding 2023-01-01 → fim

Reporta tabela 4h baseline vs 1h:
  Timeframe | VAL Sharpe | HO Sharpe | trades 17m | win% | MaxDD | final $1k
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

# ----------------------------------------------------------------- params 1h
TIMEFRAME_MIN = 60
HORIZON_MID = 48           # 48h
HORIZON_LONG = 72          # 72h
ATR_MULT = 3.0
COST = 0.0015
THRESHOLD = 0.35
NO_BEAR = -0.05
BARS_PER_DAY = 24
BARS_30D = 720             # 30 × 24
RETRAIN_EVERY_BARS = 90 * BARS_PER_DAY  # 2160
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

PROBAS_CACHE_1H = ROOT / "data" / "walk_forward_probas_1h.parquet"
PROBAS_CACHE_4H = ROOT / "data" / "walk_forward_probas.parquet"  # já existe (exp_a1)


# ----------------------------------------------------------------- build matrix
def build_matrix(horizon_bars: int, timeframe_min: int) -> tuple[pd.DataFrame, list[str]]:
    df = feat.build_v2_from_parquets(timeframe_min=timeframe_min, lag=1).drop_nulls(subset=["atr_14"])
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


# ----------------------------------------------------------------- walk-forward probas
def generate_walk_forward_probas_1h() -> pd.DataFrame:
    print(">>> [1h] construindo matrizes (h=48, h=72) ...")
    t0 = time.time()
    mat_mid, fc_mid = build_matrix(HORIZON_MID, TIMEFRAME_MIN)
    mat_long, fc_long = build_matrix(HORIZON_LONG, TIMEFRAME_MIN)
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
    start_pos = int(mat_mid["open_time"].searchsorted(start_ms))
    n_bars = len(mat_mid)
    print(f">>> [1h] simulação inicia em pos={start_pos} dt={mat_mid.iloc[start_pos]['dt']}  total={n_bars:,} bars")

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
                    print(f"  [retreino 1h] bar={i} dt={mat_mid.iloc[i]['dt'].strftime('%Y-%m-%d')} ({elapsed:.0f}s)")

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
    PROBAS_CACHE_1H.parent.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(out).write_parquet(PROBAS_CACHE_1H)
    print(f">>> [1h] probas salvas em {PROBAS_CACHE_1H} ({len(out):,} rows)")
    return out


# ----------------------------------------------------------------- simulate (MID-only, no_bear)
def simulate(probas: pd.DataFrame, horizon_mid: int, thr: float, no_bear: float | None) -> dict:
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

        # entry: MID rule only
        if not in_pos and not (np.isnan(pm[i]) or np.isnan(atrs[i])):
            sig = pm[i] > thr
            if no_bear is not None and not np.isnan(ret_30d[i]):
                if ret_30d[i] < no_bear:
                    sig = False
            if sig:
                entry_idx = i
                entry_px = closes[i]
                target_px = entry_px + ATR_MULT * atrs[i]
                stop_px = entry_px - ATR_MULT * atrs[i]
                expiry_idx = i + horizon_mid
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
    return {"equity": equity, "n_trades": len(trades), "trades": trades, "final_capital": capital}


def metrics(equity: np.ndarray, trades: list[dict], probas: pd.DataFrame,
            segment: str, bars_per_year: int) -> dict:
    dts = probas["dt"].values
    if segment == "VAL":
        mask = (dts <= np.datetime64(VAL_END))
    elif segment == "HOLDOUT":
        mask = (dts >= np.datetime64(HOLDOUT_START))
    else:
        mask = np.ones(len(dts), dtype=bool)
    seg = equity[mask]
    n_trades_seg = 0
    wins = 0
    for t in trades:
        ei = t["entry_idx"]
        if segment == "VAL" and probas["dt"].iloc[ei] <= VAL_END:
            n_trades_seg += 1
            if t["net_ret"] > 0:
                wins += 1
        elif segment == "HOLDOUT" and probas["dt"].iloc[ei] >= HOLDOUT_START:
            n_trades_seg += 1
            if t["net_ret"] > 0:
                wins += 1
        elif segment == "ALL":
            n_trades_seg += 1
            if t["net_ret"] > 0:
                wins += 1

    if len(seg) < 30:
        return {"sharpe": np.nan, "max_dd": np.nan, "ret_total": np.nan, "final": np.nan,
                "n_trades": n_trades_seg, "win_rate": np.nan}

    rets = np.diff(seg) / seg[:-1]
    rets = rets[~np.isnan(rets)]
    if len(rets) < 30 or rets.std() == 0:
        sharpe = 0.0
    else:
        sharpe = float(rets.mean() / rets.std() * np.sqrt(bars_per_year))
    peak = np.maximum.accumulate(seg)
    max_dd = float((seg / peak - 1).min())
    ret_total = float(seg[-1] / seg[0] - 1)
    win_rate = wins / n_trades_seg if n_trades_seg > 0 else np.nan
    return {
        "sharpe": sharpe, "max_dd": max_dd, "ret_total": ret_total,
        "final": float(seg[-1]), "n_trades": n_trades_seg, "win_rate": win_rate,
    }


# ----------------------------------------------------------------- run timeframe
def evaluate_timeframe(label: str, probas: pd.DataFrame, horizon_mid: int, bars_per_year: int) -> dict:
    valid = probas["proba_mid"].notna()
    probas_v = probas[valid].reset_index(drop=True)
    print(f">>> [{label}] probas válidas: {len(probas_v):,} bars  período {probas_v['dt'].iloc[0]} → {probas_v['dt'].iloc[-1]}")
    sim = simulate(probas_v, horizon_mid=horizon_mid, thr=THRESHOLD, no_bear=NO_BEAR)
    val_m = metrics(sim["equity"], sim["trades"], probas_v, "VAL", bars_per_year)
    ho_m = metrics(sim["equity"], sim["trades"], probas_v, "HOLDOUT", bars_per_year)
    all_m = metrics(sim["equity"], sim["trades"], probas_v, "ALL", bars_per_year)
    return {
        "label": label,
        "val": val_m, "ho": ho_m, "all": all_m,
        "final_capital": sim["final_capital"],
        "total_trades": sim["n_trades"],
    }


# ----------------------------------------------------------------- main
def main() -> None:
    # 1h
    if PROBAS_CACHE_1H.exists():
        print(f">>> usando cache 1h {PROBAS_CACHE_1H}")
        probas_1h = pl.read_parquet(PROBAS_CACHE_1H).to_pandas()
        probas_1h["dt"] = pd.to_datetime(probas_1h["dt"], utc=True)
    else:
        probas_1h = generate_walk_forward_probas_1h()
        probas_1h["dt"] = pd.to_datetime(probas_1h["dt"], utc=True)

    # 4h baseline (do cache do exp_a1)
    if not PROBAS_CACHE_4H.exists():
        print(f"!!! cache 4h {PROBAS_CACHE_4H} não encontrado — rode exp_a1_threshold_search.py antes")
        sys.exit(1)
    probas_4h = pl.read_parquet(PROBAS_CACHE_4H).to_pandas()
    probas_4h["dt"] = pd.to_datetime(probas_4h["dt"], utc=True)

    bars_per_year_1h = 24 * 365
    bars_per_year_4h = 6 * 365

    res_1h = evaluate_timeframe("1h", probas_1h, horizon_mid=HORIZON_MID, bars_per_year=bars_per_year_1h)
    res_4h = evaluate_timeframe("4h", probas_4h, horizon_mid=12, bars_per_year=bars_per_year_4h)

    # ----------------- print
    def fmt_pct(x):
        return f"{100*x:+.1f}%" if x is not None and not (isinstance(x, float) and np.isnan(x)) else "n/d"

    def fmt_num(x, prec=2):
        return f"{x:.{prec}f}" if x is not None and not (isinstance(x, float) and np.isnan(x)) else "n/d"

    print("\n" + "=" * 110)
    print(" RESULTADO — Timeframe 1h vs 4h (MID rule, thr=0.35, no_bear=-0.05, COST=0.0015)")
    print("=" * 110)
    header = f"{'TF':<6}{'VAL Shp':>10}{'HO Shp':>10}{'trades VAL':>14}{'trades HO':>12}{'win% HO':>12}{'MaxDD HO':>12}{'final $1k':>14}"
    print(header)
    print("-" * 110)
    for r in (res_4h, res_1h):
        line = (
            f"{r['label']:<6}"
            f"{fmt_num(r['val']['sharpe']):>10}"
            f"{fmt_num(r['ho']['sharpe']):>10}"
            f"{r['val']['n_trades']:>14d}"
            f"{r['ho']['n_trades']:>12d}"
            f"{fmt_pct(r['ho']['win_rate']):>12}"
            f"{fmt_pct(r['ho']['max_dd']):>12}"
            f"${r['final_capital']:>13,.0f}"
        )
        print(line)
    print("=" * 110)

    # decisão
    s1h_ho = res_1h["ho"]["sharpe"]
    s4h_ho = res_4h["ho"]["sharpe"]
    s1h_val = res_1h["val"]["sharpe"]
    s4h_val = res_4h["val"]["sharpe"]
    n1h_ho = res_1h["ho"]["n_trades"]
    n4h_ho = res_4h["ho"]["n_trades"]

    print("\n" + "=" * 110)
    print(" DECISÃO")
    print("=" * 110)
    print(f"  HO Sharpe: 1h={s1h_ho:+.2f} vs 4h={s4h_ho:+.2f}  (delta {s1h_ho - s4h_ho:+.2f})")
    print(f"  HO trades: 1h={n1h_ho} vs 4h={n4h_ho}  (ratio {n1h_ho / max(1, n4h_ho):.2f}x)")
    print(f"  VAL Sharpe: 1h={s1h_val:+.2f} vs 4h={s4h_val:+.2f}")
    if s1h_ho >= s4h_ho * 0.9 and n1h_ho >= n4h_ho * 1.5:
        print("  >>> 1h MELHOR — Sharpe similar/superior com >50% mais trades. Vale migrar.")
    elif s1h_ho >= s4h_ho * 0.7:
        print("  >>> 1h SIMILAR — Sharpe próximo. Avaliar custo/benefício de operacional 1h.")
    elif s1h_ho < 0:
        print("  >>> 1h PIOR (negativo). Ficar no 4h baseline.")
    else:
        print("  >>> 1h PIOR — Sharpe degrada significativamente vs 4h. Ficar no 4h baseline.")
    print("=" * 110)


if __name__ == "__main__":
    main()
