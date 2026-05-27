"""exp_e3_retrain_freq — varre frequencia de retreino walk-forward.

Hipotese: BTC tem regime shifts; retreinar mais frequente (mensal) pode adaptar
melhor. Trade-off: mais retreinos = mais compute e mais perda de barras iniciais
por necessidade de label realizado.

Setup (winner do A1):
- Rule: MID-only
- thr_mid: 0.35
- no_bear: -0.05 (ret_30d >= -5%)
- COST: 0.0015
- FULL sizing (1x capital)
- Expanding window (treina em tudo antes), apenas freq de retreino varia
- uniqueness weights

Frequencias:
- 30 dias (180 bars 4h) — mensal
- 60 dias (360 bars) — bi-mensal
- 90 dias (540 bars) — quarterly (baseline atual)
- 180 dias (1080 bars) — semestral

Pra cada freq:
1. Walk-forward gerando proba_mid em cada bar (MID-only — long nao necessario)
2. Simula trades com regra MID + NO_BEAR
3. Reporta Sharpe VAL / HOLDOUT, trades, final $1k

Output: tabela comparativa + recomendacao.
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

# ----------------------------------------------------------------- params
TIMEFRAME_MIN = 240
HORIZON_MID = 12
ATR_MULT = 3.0
COST = 0.0015
BARS_PER_DAY = 6
START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)
VAL_END = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
HOLDOUT_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
INITIAL_CAPITAL = 1000.0

# Winner config (A1)
THR_MID = 0.35
NO_BEAR = -0.05
BARS_30D = 30 * BARS_PER_DAY  # 180

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

FREQS_DAYS = [30, 60, 90, 180]


# ----------------------------------------------------------------- build matrix
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


# ----------------------------------------------------------------- walk-forward MID-only
def walk_forward_probas(mat_mid: pd.DataFrame, fc_mid: list[str], retrain_every: int) -> tuple[pd.DataFrame, int]:
    """Walk-forward gerando proba_mid em cada bar; retorna df + n_retrains."""
    fc_mid_arr = mat_mid[fc_mid].to_numpy()
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

    model_mid: lgb.Booster | None = None
    last_train_idx = -10**9
    n_retrains = 0
    probas_mid = np.full(n_bars, np.nan)

    t0 = time.time()
    for i in range(start_pos, n_bars):
        if model_mid is None or (i - last_train_idx) >= retrain_every:
            cut_mid = i - HORIZON_MID
            if cut_mid > 500:
                X_tr = fc_mid_arr[:cut_mid]
                y_tr = y_mid_arr[:cut_mid]
                w_tr = w_mid_arr[:cut_mid]
                model_mid = lgb.train(
                    LGB_PARAMS,
                    lgb.Dataset(X_tr, y_tr, weight=w_tr),
                    num_boost_round=N_ROUNDS,
                )
                last_train_idx = i
                n_retrains += 1

        if model_mid is not None:
            probas_mid[i] = float(model_mid.predict(fc_mid_arr[i : i + 1])[0])

    # ret_30d pro filtro NO_BEAR
    ret_30d = np.full(n_bars, np.nan)
    for i in range(BARS_30D, n_bars):
        if closes[i - BARS_30D] > 0:
            ret_30d[i] = closes[i] / closes[i - BARS_30D] - 1

    dt = pd.to_datetime(open_times, unit="ms", utc=True)
    out = pd.DataFrame({
        "open_time": open_times,
        "dt": dt,
        "open": mat_mid["open"].values,
        "high": highs,
        "low": lows,
        "close": closes,
        "atr": atrs,
        "proba_mid": probas_mid,
        "ret_30d": ret_30d,
    })
    elapsed = time.time() - t0
    print(f"    walk-forward: {n_retrains} retreinos, {elapsed:.0f}s")
    return out, n_retrains


# ----------------------------------------------------------------- simulate MID-only FULL sizing
def simulate(probas: pd.DataFrame) -> dict:
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

        # entry MID-only
        if not in_pos and not (np.isnan(pm[i]) or np.isnan(atrs[i])):
            sig = pm[i] > THR_MID
            if not np.isnan(ret_30d[i]) and ret_30d[i] < NO_BEAR:
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

    if in_pos:
        net = closes[-1] / entry_px - 1 - COST
        capital *= (1 + net)
        trades.append({"entry_idx": entry_idx, "exit_idx": n - 1, "net_ret": net})
        equity[-1] = capital

    return {"equity": equity, "n_trades": len(trades), "trades": trades, "final_capital": capital}


# ----------------------------------------------------------------- metrics
def metrics(equity: np.ndarray, dt_arr: np.ndarray, segment: str) -> dict:
    bars_per_year = 6 * 365
    val_end_ns = np.datetime64(VAL_END.replace(tzinfo=None), "ns")
    ho_start_ns = np.datetime64(HOLDOUT_START.replace(tzinfo=None), "ns")
    if segment == "VAL":
        mask = dt_arr <= val_end_ns
    elif segment == "HOLDOUT":
        mask = dt_arr >= ho_start_ns
    else:
        mask = np.ones(len(dt_arr), dtype=bool)
    seg = equity[mask]
    if len(seg) < 30:
        return {"sharpe": np.nan, "max_dd": np.nan, "n_bars": len(seg), "ret_total": np.nan, "final": np.nan}
    rets = np.diff(seg) / seg[:-1]
    rets = rets[~np.isnan(rets)]
    if len(rets) < 30 or rets.std() == 0:
        sharpe = 0.0
    else:
        sharpe = float(rets.mean() / rets.std() * np.sqrt(bars_per_year))
    peak = np.maximum.accumulate(seg)
    max_dd = float((seg / peak - 1).min())
    ret_total = float(seg[-1] / seg[0] - 1)
    return {
        "sharpe": sharpe,
        "max_dd": max_dd,
        "n_bars": len(seg),
        "ret_total": ret_total,
        "final": float(seg[-1]),
    }


# ----------------------------------------------------------------- main
def main() -> None:
    print(">>> building feature matrix (h=12) ...")
    t0 = time.time()
    mat_mid, fc_mid = build_matrix(HORIZON_MID)
    print(f"    mid: {len(mat_mid):,} rows  ({time.time() - t0:.1f}s)")
    print(f">>> running E3: retrain frequency sweep")
    print(f"    config: rule=MID, thr={THR_MID}, no_bear={NO_BEAR}, COST={COST}, sizing=FULL, uniqueness=ON")
    print(f"    freqs: {FREQS_DAYS} dias")

    results = []
    for days in FREQS_DAYS:
        retrain_every = days * BARS_PER_DAY
        print(f"\n>>> freq = {days} dias ({retrain_every} bars 4h)")
        probas, n_retrains = walk_forward_probas(mat_mid, fc_mid, retrain_every)

        # filtra apenas bars com proba valida (depois de pulo inicial p/ primeiro modelo)
        valid_mask = probas["proba_mid"].notna()
        probas_v = probas[valid_mask].reset_index(drop=True)

        sim = simulate(probas_v)
        # converte dt p/ datetime64 naive (UTC) p/ comparacao
        dt_arr = probas_v["dt"].dt.tz_convert("UTC").dt.tz_localize(None).values.astype("datetime64[ns]")
        val_m = metrics(sim["equity"], dt_arr, "VAL")
        ho_m = metrics(sim["equity"], dt_arr, "HOLDOUT")
        full_m = metrics(sim["equity"], dt_arr, "FULL")

        # contagem de trades por segmento
        trades_df = pd.DataFrame(sim["trades"])
        val_end_ns = np.datetime64(VAL_END.replace(tzinfo=None), "ns")
        ho_start_ns = np.datetime64(HOLDOUT_START.replace(tzinfo=None), "ns")
        if len(trades_df):
            entry_dts = dt_arr[trades_df["entry_idx"].to_numpy()]
            n_trades_val = int(np.sum(entry_dts <= val_end_ns))
            n_trades_ho = int(np.sum(entry_dts >= ho_start_ns))
        else:
            n_trades_val = n_trades_ho = 0

        results.append({
            "freq_days": days,
            "n_retrains": n_retrains,
            "val_sharpe": val_m["sharpe"],
            "ho_sharpe": ho_m["sharpe"],
            "val_ret": val_m["ret_total"],
            "ho_ret": ho_m["ret_total"],
            "val_dd": val_m["max_dd"],
            "ho_dd": ho_m["max_dd"],
            "n_trades_total": sim["n_trades"],
            "n_trades_val": n_trades_val,
            "n_trades_ho": n_trades_ho,
            "final_1k": sim["final_capital"],
        })
        print(
            f"    VAL sharpe={val_m['sharpe']:+.2f}  HO sharpe={ho_m['sharpe']:+.2f}  "
            f"trades={sim['n_trades']} (val {n_trades_val}, ho {n_trades_ho})  "
            f"final ${sim['final_capital']:,.0f}"
        )

    df = pd.DataFrame(results)
    out_csv = ROOT / "data" / "exp_e3_retrain_freq_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n>>> resultados salvos em {out_csv}")

    print("\n" + "=" * 100)
    print(" RESULTADOS E3 — varredura de frequencia de retreino (MID-only, thr=0.35, no_bear=-0.05)")
    print("=" * 100)
    header = f" {'freq':>6s} | {'n_retr':>6s} | {'VAL Sh':>8s} | {'HO Sh':>8s} | {'trades':>7s} | {'val/ho':>9s} | {'VAL ret':>9s} | {'HO ret':>9s} | {'final $1k':>10s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f" {r['freq_days']:>4d}d | {r['n_retrains']:>6d} | "
            f"{r['val_sharpe']:>+8.2f} | {r['ho_sharpe']:>+8.2f} | "
            f"{r['n_trades_total']:>7d} | {r['n_trades_val']:>3d}/{r['n_trades_ho']:>3d}    | "
            f"{100*r['val_ret']:>+8.1f}% | {100*r['ho_ret']:>+8.1f}% | ${r['final_1k']:>9,.0f}"
        )
    print("=" * 100)

    # recomendacao
    baseline = next(r for r in results if r["freq_days"] == 90)
    best_ho = max(results, key=lambda r: r["ho_sharpe"])
    print("\n>>> Analise:")
    print(f"  Baseline (90d quarterly): HO sharpe {baseline['ho_sharpe']:+.2f}, final ${baseline['final_1k']:,.0f}")
    print(f"  Melhor HO sharpe: freq={best_ho['freq_days']}d sharpe={best_ho['ho_sharpe']:+.2f} final=${best_ho['final_1k']:,.0f}")
    if best_ho["freq_days"] == 90:
        print("  >>> RECOMENDACAO: manter 90d (baseline ja eh otimo).")
    else:
        delta_sh = best_ho["ho_sharpe"] - baseline["ho_sharpe"]
        delta_final = best_ho["final_1k"] - baseline["final_1k"]
        improve = delta_sh / max(0.01, abs(baseline["ho_sharpe"])) * 100
        print(f"  >>> Delta HO Sharpe: {delta_sh:+.2f} ({improve:+.0f}%)  Delta final: ${delta_final:+,.0f}")
        if delta_sh >= 0.15 and abs(delta_final) >= 50:
            print(f"  >>> RECOMENDACAO: trocar para {best_ho['freq_days']}d — melhora material em HOLDOUT.")
        elif delta_sh > 0:
            print(f"  >>> RECOMENDACAO: marginal — pode trocar para {best_ho['freq_days']}d mas ganho pequeno.")
        else:
            print(f"  >>> RECOMENDACAO: manter 90d — diferenca dentro do ruido.")


if __name__ == "__main__":
    main()
