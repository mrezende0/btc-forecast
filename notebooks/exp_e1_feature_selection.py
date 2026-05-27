"""exp_e1_feature_selection — Feature selection via LightGBM gain importance.

Hipótese: o modelo MID (h=12) tem features fracas que adicionam ruído. Dropar o
bottom 30-40% por importance pode melhorar estabilidade e reduzir overfit.

Setup FIXO (não revisita):
- ENSEMBLE_RULE="MID", thr=0.35, no_bear=-0.05, COST=0.0015, FULL sizing
- Walk-forward expanding quarterly (RETRAIN_EVERY_BARS = 90*6 = 540)
- Purge HORIZON=12
- BTC apenas, timeframe 4h
- Uniqueness weights ON

Pipeline:
1. Constrói matriz MID (h=12) uma vez.
2. Roda walk-forward com TODAS as features. Em cada retreino, salva
   feature_importance (gain). Acumula probas pra avaliar baseline.
3. Agrega importance: média do gain por feature ao longo dos folds (normalizada
   por fold pra fold de tamanho diferente não dominar).
4. Pra cada config {all, top40, top30, top20, top15}:
   - Reroda walk-forward com SUBSET de features (mesma janela, mesmas datas
     de retreino).
   - Simula trades MID + thr=0.35 + no_bear=-0.05.
   - Mede Sharpe/PSR/MaxDD/n_trades/$final em VAL e HOLDOUT.
5. Reporta tabela final.

Conclusão: existe sweet spot onde menos features = mais Sharpe?
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

# ----------------------------------------------------------------- params fixos
TIMEFRAME_MIN = 240
HORIZON_MID = 12
ATR_MULT = 3.0
COST = 0.0015
BARS_PER_DAY = 6
RETRAIN_EVERY_BARS = 90 * BARS_PER_DAY  # ~quarterly expanding
START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)
VAL_END = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
HOLDOUT_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
INITIAL_CAPITAL = 1000.0

THR_MID = 0.35
NO_BEAR = -0.05
BARS_30D = 180  # 30d * 6 bars 4h

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

CACHE_DIR = ROOT / "data"
IMPORTANCE_CACHE = CACHE_DIR / "exp_e1_importance.csv"


# ----------------------------------------------------------------- step 1: matriz
def build_matrix() -> tuple[pd.DataFrame, list[str]]:
    df = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME_MIN, lag=1).drop_nulls(subset=["atr_14"])
    labeled = lab.triple_barrier(df, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=HORIZON_MID)
    labeled = lab.attach_uniqueness(labeled, horizon_bars=HORIZON_MID)
    labeled = labeled.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y"))
    fc = [
        c for c in labeled.columns
        if c not in feat.LAG_SAFE_EXCLUDE
        and c not in {"label", "hit_bar", "barrier_ret", "upper_px", "lower_px", "y", "uniqueness_weight"}
    ]
    base_cols = ["open_time", "open", "high", "low", "close", "y", "uniqueness_weight"]
    keep = base_cols + [c for c in fc if c not in base_cols]
    mat = labeled.select(keep).drop_nulls(subset=fc + ["y"]).to_pandas()
    return mat, fc


# ----------------------------------------------------------------- step 2: WF
def walk_forward_probas(
    mat: pd.DataFrame,
    feature_cols: list[str],
    collect_importance: bool = False,
) -> tuple[np.ndarray, list[pd.Series]]:
    """Walk-forward MID. Retorna (probas, [importance_per_fold] se collect)."""
    fc_arr = mat[feature_cols].to_numpy()
    open_times = mat["open_time"].to_numpy()
    y_arr = mat["y"].to_numpy()
    w_arr = mat["uniqueness_weight"].to_numpy()
    n = len(mat)

    start_ms = int(START_DATE.timestamp() * 1000)
    start_pos = int(np.searchsorted(open_times, start_ms))

    model: lgb.Booster | None = None
    last_train_idx = -10**9

    probas = np.full(n, np.nan)
    importances: list[pd.Series] = []

    for i in range(start_pos, n):
        if model is None or (i - last_train_idx) >= RETRAIN_EVERY_BARS:
            cut = i - HORIZON_MID
            if cut > 500:
                X_tr = fc_arr[:cut]
                y_tr = y_arr[:cut]
                w_tr = w_arr[:cut]
                model = lgb.train(
                    LGB_PARAMS,
                    lgb.Dataset(X_tr, y_tr, weight=w_tr, feature_name=feature_cols),
                    num_boost_round=N_ROUNDS,
                )
                last_train_idx = i
                if collect_importance:
                    imp = pd.Series(
                        model.feature_importance(importance_type="gain"),
                        index=feature_cols,
                    )
                    importances.append(imp)
        if model is not None:
            probas[i] = float(model.predict(fc_arr[i : i + 1])[0])

    return probas, importances


# ----------------------------------------------------------------- step 3: simulate
def simulate_mid(
    mat: pd.DataFrame,
    probas: np.ndarray,
    thr_mid: float,
    no_bear: float | None,
) -> dict:
    closes = mat["close"].to_numpy()
    highs = mat["high"].to_numpy()
    lows = mat["low"].to_numpy()
    atrs = mat["atr_14"].to_numpy()
    n = len(mat)

    # ret_30d
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
    trades: list[dict] = []

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
                capital *= 1 + net
                trades.append({"entry_idx": entry_idx, "exit_idx": i, "net_ret": net})
                in_pos = False

        if not in_pos and not (np.isnan(probas[i]) or np.isnan(atrs[i])):
            sig = probas[i] > thr_mid
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

    if in_pos:
        net = closes[-1] / entry_px - 1 - COST
        capital *= 1 + net
        trades.append({"entry_idx": entry_idx, "exit_idx": n - 1, "net_ret": net})
        equity[-1] = capital

    return {"equity": equity, "trades": trades, "final_capital": capital}


def metrics(equity: np.ndarray, dts: np.ndarray, segment: str) -> dict:
    bars_per_year = 6 * 365
    val_end_np = np.datetime64(VAL_END.replace(tzinfo=None))
    ho_start_np = np.datetime64(HOLDOUT_START.replace(tzinfo=None))
    if segment == "VAL":
        mask = dts <= val_end_np
    elif segment == "HOLDOUT":
        mask = dts >= ho_start_np
    else:
        mask = np.ones(len(dts), dtype=bool)
    seg = equity[mask]
    if len(seg) < 30:
        return {"sharpe": np.nan, "psr_0": np.nan, "max_dd": np.nan, "ret_total": np.nan, "final": np.nan}
    rets = np.diff(seg) / seg[:-1]
    rets = rets[~np.isnan(rets)]
    if len(rets) < 30 or rets.std() == 0:
        return {"sharpe": 0.0, "psr_0": 0.5, "max_dd": 0.0, "ret_total": float(seg[-1] / seg[0] - 1), "final": float(seg[-1])}
    sharpe = float(rets.mean() / rets.std() * np.sqrt(bars_per_year))
    from scipy.stats import norm
    sr_pb = rets.mean() / rets.std()
    skew = float(pd.Series(rets).skew())
    kurt = float(pd.Series(rets).kurtosis())
    denom = np.sqrt(max(1e-12, (1 - skew * sr_pb + kurt / 4 * sr_pb ** 2) / (len(rets) - 1)))
    psr_0 = float(norm.cdf(sr_pb / denom))
    peak = np.maximum.accumulate(seg)
    max_dd = float((seg / peak - 1).min())
    ret_total = float(seg[-1] / seg[0] - 1)
    return {"sharpe": sharpe, "psr_0": psr_0, "max_dd": max_dd, "ret_total": ret_total, "final": float(seg[-1])}


# ----------------------------------------------------------------- main
def main() -> None:
    print(">>> step 1: build matrix (h=12) ...")
    t0 = time.time()
    mat, fc_all = build_matrix()
    mat["dt"] = pd.to_datetime(mat["open_time"], unit="ms")
    print(f"  rows={len(mat):,}  features={len(fc_all)}  ({time.time() - t0:.1f}s)")
    print(f"  período: {mat['dt'].iloc[0]} → {mat['dt'].iloc[-1]}")

    # step 2: WF baseline + importance
    print("\n>>> step 2: walk-forward com TODAS as features (coleta importance) ...")
    t0 = time.time()
    probas_all, importances = walk_forward_probas(mat, fc_all, collect_importance=True)
    print(f"  done ({time.time() - t0:.0f}s)  folds={len(importances)}")

    # agregação: cada fold normalizado (soma=1), depois média
    imp_df = pd.concat(importances, axis=1)
    imp_norm = imp_df.div(imp_df.sum(axis=0), axis=1)  # cada fold soma 1
    imp_mean = imp_norm.mean(axis=1).sort_values(ascending=False)
    imp_mean.name = "gain_mean_norm"
    imp_mean.to_csv(IMPORTANCE_CACHE, header=True)
    print(f"\n  ranking salvo em {IMPORTANCE_CACHE}")
    print("\n  TOP-15 features (importance média normalizada):")
    print(imp_mean.head(15).to_string())
    print("\n  BOTTOM-10:")
    print(imp_mean.tail(10).to_string())

    # configs
    configs = {
        "all_features": fc_all,
        "top_40": imp_mean.head(40).index.tolist(),
        "top_30": imp_mean.head(30).index.tolist(),
        "top_20": imp_mean.head(20).index.tolist(),
        "top_15": imp_mean.head(15).index.tolist(),
    }

    dts_np = mat["dt"].values
    results = []

    # cachear baseline (já tem probas)
    sim = simulate_mid(mat, probas_all, THR_MID, NO_BEAR)
    val_m = metrics(sim["equity"], dts_np, "VAL")
    ho_m = metrics(sim["equity"], dts_np, "HOLDOUT")
    n_val = sum(1 for t in sim["trades"] if dts_np[t["entry_idx"]] <= np.datetime64(VAL_END.replace(tzinfo=None)))
    n_ho = sum(1 for t in sim["trades"] if dts_np[t["entry_idx"]] >= np.datetime64(HOLDOUT_START.replace(tzinfo=None)))
    results.append({
        "config": "all_features",
        "n_features": len(fc_all),
        "val_sharpe": val_m["sharpe"], "ho_sharpe": ho_m["sharpe"],
        "val_psr0": val_m["psr_0"], "ho_psr0": ho_m["psr_0"],
        "val_dd": val_m["max_dd"], "ho_dd": ho_m["max_dd"],
        "n_trades_val": n_val, "n_trades_ho": n_ho,
        "final_ho": ho_m["final"],
        "final_total": sim["final_capital"],
    })

    # step 4: roda cada subset
    for name, subset in configs.items():
        if name == "all_features":
            continue
        print(f"\n>>> step 4: walk-forward [{name}]  n_features={len(subset)} ...")
        t0 = time.time()
        probas, _ = walk_forward_probas(mat, subset, collect_importance=False)
        sim = simulate_mid(mat, probas, THR_MID, NO_BEAR)
        val_m = metrics(sim["equity"], dts_np, "VAL")
        ho_m = metrics(sim["equity"], dts_np, "HOLDOUT")
        n_val = sum(1 for t in sim["trades"] if dts_np[t["entry_idx"]] <= np.datetime64(VAL_END.replace(tzinfo=None)))
        n_ho = sum(1 for t in sim["trades"] if dts_np[t["entry_idx"]] >= np.datetime64(HOLDOUT_START.replace(tzinfo=None)))
        results.append({
            "config": name,
            "n_features": len(subset),
            "val_sharpe": val_m["sharpe"], "ho_sharpe": ho_m["sharpe"],
            "val_psr0": val_m["psr_0"], "ho_psr0": ho_m["psr_0"],
            "val_dd": val_m["max_dd"], "ho_dd": ho_m["max_dd"],
            "n_trades_val": n_val, "n_trades_ho": n_ho,
            "final_ho": ho_m["final"],
            "final_total": sim["final_capital"],
        })
        print(f"  done ({time.time() - t0:.0f}s)  VAL Sharpe={val_m['sharpe']:+.2f}  HO Sharpe={ho_m['sharpe']:+.2f}")

    df = pd.DataFrame(results)
    out_csv = ROOT / "data" / "exp_e1_feature_selection_results.csv"
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 105)
    print(" RESULTADOS — Feature Selection (MID, thr=0.35, no_bear=-0.05, COST=0.0015, FULL sizing)")
    print("=" * 105)
    show = df.copy()
    for c in ["val_sharpe", "ho_sharpe", "val_psr0", "ho_psr0", "val_dd", "ho_dd"]:
        show[c] = show[c].map(lambda x: f"{x:+.2f}" if not pd.isna(x) else "n/d")
    for c in ["final_ho", "final_total"]:
        show[c] = show[c].map(lambda x: f"${x:.0f}" if not pd.isna(x) else "n/d")
    print(show.to_string(index=False))
    print("=" * 105)
    print(f"\n>>> CSV: {out_csv}")

    # recomendação
    base = df.iloc[0]
    best_idx = df["ho_sharpe"].idxmax()
    best = df.iloc[best_idx]
    print("\nRECOMENDAÇÃO:")
    print(f"  baseline (all): HO Sharpe {base['ho_sharpe']:+.2f} | n={base['n_features']} | final ${base['final_ho']:.0f}")
    print(f"  melhor:        [{best['config']}] HO Sharpe {best['ho_sharpe']:+.2f} | n={best['n_features']} | final ${best['final_ho']:.0f}")
    delta = best["ho_sharpe"] - base["ho_sharpe"]
    if best["config"] == "all_features":
        print(f"  >>> NÃO compensa dropar — baseline já é o melhor.")
    elif delta >= 0.1:
        print(f"  >>> VALE dropar features. Ganho HO Sharpe Δ={delta:+.2f} com {base['n_features']-best['n_features']} features a menos.")
    elif delta >= 0:
        print(f"  >>> Ganho marginal (Δ={delta:+.2f}). Considere manter all_features pela robustez estatística.")
    else:
        print(f"  >>> NÃO compensa — baseline > best (Δ={delta:+.2f}).")


if __name__ == "__main__":
    main()
