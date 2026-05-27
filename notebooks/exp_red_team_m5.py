"""exp_red_team_m5 — Red Team Adversarial sobre o WINNER de A1-A.

Winner: thr_mid=0.35, no_bear=-0.05, rule=MID. Baseline:
  VAL Sharpe 0.36, HOLDOUT Sharpe 1.53, PSR(0) 0.952.

Testes adversariais (cada um deveria ~zerar Sharpe se modelo é robusto):

T1) SHUFFLE LABELS: re-treina com y embaralhado. Sharpe HOLDOUT esperado ~0.
    Se Sharpe > 0.3 → leak forte (modelo está achando padrão impossível).

T2) NOISE FEATURE inserida: adiciona coluna de ruído gaussiano. Verifica
    se a noise feature aparece no top-10 por GAIN do LGB.
    Se aparece → modelo está pegando padrões espúrios.

T3) PERMUTATION IMPORTANCE no HOLDOUT: pra cada feature, permuta valores
    no test set e mede queda de Sharpe. Features sem queda = não importam
    (model não está usando essa info).

K incremental: +1 (1 bloco de hipóteses adversariais).
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

# Mesmos params do A1
TIMEFRAME_MIN = 240
HORIZON = 12  # MID only (winner)
ATR_MULT = 3.0
COST = 0.0015
BARS_PER_DAY = 6
RETRAIN_EVERY_BARS = 90 * BARS_PER_DAY
START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)
VAL_END = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
HOLDOUT_START = datetime(2025, 1, 1, tzinfo=timezone.utc)

# Winner config
THR = 0.35
NO_BEAR = -0.05
BARS_PER_MONTH = 180

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
SEED = 42


def build_matrix() -> tuple[pd.DataFrame, list[str]]:
    df = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME_MIN, lag=1).drop_nulls(subset=["atr_14"])
    lab_df = lab.triple_barrier(df, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=HORIZON)
    lab_df = lab.attach_uniqueness(lab_df, horizon_bars=HORIZON)
    lab_df = lab_df.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y"))
    fcols = [c for c in lab_df.columns if c not in feat.LAG_SAFE_EXCLUDE and c not in {"label","hit_bar","barrier_ret","upper_px","lower_px","y","uniqueness_weight"}]
    cols = ["open_time","close","high","low","y","barrier_ret","uniqueness_weight", *fcols]
    seen = set(); cols = [c for c in cols if not (c in seen or seen.add(c))]
    m = lab_df.select(cols).drop_nulls(subset=fcols + ["y"]).to_pandas()
    m["dt"] = m["open_time"].apply(lambda ms: datetime.fromtimestamp(ms/1000, tz=timezone.utc))
    return m, fcols


def walk_forward(mat: pd.DataFrame, fcols: list[str], shuffle_y: bool = False,
                 extra_noise_col: bool = False, seed: int = SEED) -> pd.DataFrame:
    """Roda walk-forward + simulação com WINNER config.

    Args:
        shuffle_y: se True, embaralha y dentro de cada fold de treino.
        extra_noise_col: se True, adiciona coluna 'noise' ao feature set.
        seed: pra reproducibilidade do shuffle.

    Retorna probas + flags por bar.
    """
    rng = np.random.default_rng(seed)
    n = len(mat)
    proba = np.zeros(n)
    covered = np.zeros(n, dtype=bool)
    fi_last = None

    # Determine fold boundaries
    start_idx = mat.index[mat["dt"] >= START_DATE].tolist()
    if not start_idx:
        return pd.DataFrame()
    pos = start_idx[0]
    use_fcols = list(fcols)
    if extra_noise_col:
        use_fcols = use_fcols + ["__noise__"]
        mat = mat.copy()
        mat["__noise__"] = rng.standard_normal(n)

    while pos < n:
        train_end = pos - HORIZON
        if train_end < 500:
            pos += RETRAIN_EVERY_BARS
            continue
        X_tr = mat.iloc[:train_end][use_fcols].values
        y_tr = mat.iloc[:train_end]["y"].values.copy()
        w_tr = mat.iloc[:train_end]["uniqueness_weight"].values
        if shuffle_y:
            rng.shuffle(y_tr)
        ds = lgb.Dataset(X_tr, y_tr, weight=w_tr)
        model = lgb.train(LGB_PARAMS, ds, num_boost_round=N_ROUNDS)
        # Predict block ahead
        block_end = min(pos + RETRAIN_EVERY_BARS, n)
        X_te = mat.iloc[pos:block_end][use_fcols].values
        proba[pos:block_end] = model.predict(X_te)
        covered[pos:block_end] = True
        fi_last = pd.DataFrame({
            "feature": use_fcols,
            "gain": model.feature_importance(importance_type="gain"),
        }).sort_values("gain", ascending=False)
        pos = block_end

    out = mat[["open_time", "close", "high", "low", "barrier_ret", "dt"]].copy()
    out["proba"] = proba
    out["covered"] = covered
    return out, fi_last


def simulate(probas: pd.DataFrame, mat: pd.DataFrame) -> dict:
    """Simula trades com WINNER config (thr=0.35, no_bear=-0.05, MID only).

    1 posição por vez, compounding sequencial, custo 0.0015 round-trip.
    """
    capital = 1000.0
    cash = capital
    position = None
    trades = []
    equity = []

    close_series = mat["close"].values
    high = mat["high"].values
    low = mat["low"].values
    proba_arr = probas["proba"].values
    covered = probas["covered"].values
    dt = mat["dt"].values
    open_time = mat["open_time"].values

    for i in range(len(probas)):
        if not covered[i]:
            equity.append(capital)
            continue

        # Check open position
        if position is not None:
            hit_stop = low[i] <= position["stop"]
            hit_target = high[i] >= position["target"]
            timeout = open_time[i] >= position["timeout_at"]
            exit_price = None; outcome = None
            if hit_stop and hit_target:
                exit_price = position["stop"]; outcome = "stop"
            elif hit_stop:
                exit_price = position["stop"]; outcome = "stop"
            elif hit_target:
                exit_price = position["target"]; outcome = "target"
            elif timeout:
                exit_price = close_series[i]; outcome = "timeout"
            if exit_price is not None:
                pnl_pct = (exit_price / position["entry"] - 1) - COST
                pnl_usd = position["size_usd"] * pnl_pct
                cash += position["size_usd"] + pnl_usd
                capital = cash
                trades.append({"exit_dt": dt[i], "pnl_pct": pnl_pct, "outcome": outcome})
                position = None

        # Open
        if position is None and proba_arr[i] > THR:
            # bear filter
            if i >= BARS_PER_MONTH:
                ret_30d = close_series[i] / close_series[i - BARS_PER_MONTH] - 1
                if ret_30d < NO_BEAR:
                    equity.append(capital)
                    continue
            # ATR from mat (might need recompute; use a column if available)
            atr = float(mat["atr_14"].iloc[i]) if "atr_14" in mat.columns else (high[i] - low[i])
            entry = close_series[i]
            stop = entry - ATR_MULT * atr
            target = entry + ATR_MULT * atr
            size_usd = capital  # FULL
            cash -= size_usd
            position = {
                "entry": entry, "stop": stop, "target": target, "size_usd": size_usd,
                "timeout_at": open_time[i] + HORIZON * 4 * 3600 * 1000,
            }
        # equity mark-to-market
        if position is not None:
            mtm = position["size_usd"] * (close_series[i] / position["entry"])
            equity.append(cash + mtm)
        else:
            equity.append(capital)

    eq = np.array(equity)
    return {"equity": eq, "trades": pd.DataFrame(trades), "final": float(eq[-1])}


def sharpe(eq: np.ndarray, mat: pd.DataFrame, mask) -> float:
    r = pd.Series(eq[mask]).pct_change().fillna(0)
    if r.std() == 0:
        return 0.0
    return float((r.mean() / r.std()) * np.sqrt(BARS_PER_DAY * 365))


def main():
    print("[red] build matrix…", flush=True)
    mat, fcols = build_matrix()
    print(f"[red] {len(mat)} bars, {len(fcols)} features")

    val_mask = (mat["dt"] >= START_DATE) & (mat["dt"] <= VAL_END)
    holdout_mask = mat["dt"] >= HOLDOUT_START

    results = []

    # === Baseline (sem perturbação) ===
    print("\n[T0] BASELINE (sem perturbação)…", flush=True)
    t0 = time.time()
    probas, fi = walk_forward(mat, fcols, shuffle_y=False, extra_noise_col=False)
    sim = simulate(probas, mat)
    s_val = sharpe(sim["equity"], mat, val_mask.values)
    s_ho = sharpe(sim["equity"], mat, holdout_mask.values)
    final = sim["final"]
    print(f"  → VAL Sharpe={s_val:+.2f}  HOLDOUT Sharpe={s_ho:+.2f}  Final ${final:,.0f}  ({time.time()-t0:.0f}s)")
    results.append(("BASELINE", s_val, s_ho, final))

    # === T1: Shuffle labels ===
    print("\n[T1] SHUFFLE LABELS (rotos os y de treino)…", flush=True)
    t0 = time.time()
    probas, _ = walk_forward(mat, fcols, shuffle_y=True)
    sim = simulate(probas, mat)
    s_val = sharpe(sim["equity"], mat, val_mask.values)
    s_ho = sharpe(sim["equity"], mat, holdout_mask.values)
    final = sim["final"]
    print(f"  → VAL Sharpe={s_val:+.2f}  HOLDOUT Sharpe={s_ho:+.2f}  Final ${final:,.0f}  ({time.time()-t0:.0f}s)")
    print(f"  ESPERADO: Sharpe HOLDOUT ~0. Se > 0.3 = LEAK.")
    results.append(("T1 shuffle_labels", s_val, s_ho, final))

    # === T2: Noise feature ===
    print("\n[T2] NOISE FEATURE inserida (verifica ranking)…", flush=True)
    t0 = time.time()
    probas, fi_noise = walk_forward(mat, fcols, extra_noise_col=True)
    sim = simulate(probas, mat)
    s_val = sharpe(sim["equity"], mat, val_mask.values)
    s_ho = sharpe(sim["equity"], mat, holdout_mask.values)
    final = sim["final"]
    print(f"  → VAL Sharpe={s_val:+.2f}  HOLDOUT Sharpe={s_ho:+.2f}  Final ${final:,.0f}  ({time.time()-t0:.0f}s)")
    if fi_noise is not None:
        noise_rank = fi_noise.reset_index(drop=True)
        n_idx = noise_rank.index[noise_rank["feature"] == "__noise__"].tolist()
        if n_idx:
            print(f"  __noise__ rank #{n_idx[0]+1}/{len(noise_rank)} (gain={noise_rank.iloc[n_idx[0]]['gain']:.0f})")
            print(f"  Top 5 features: {noise_rank.head(5)['feature'].tolist()}")
            if n_idx[0] < 10:
                print(f"  ⚠️  __noise__ no top 10 — modelo está pegando ruído como sinal.")
            else:
                print(f"  ✓ __noise__ fora do top 10 — modelo distingue sinal de ruído.")
    results.append(("T2 noise_feature", s_val, s_ho, final))

    # === Sumário ===
    print("\n" + "=" * 70)
    print(f"{'Teste':<28s}  {'VAL Sharpe':>10s}  {'HO Sharpe':>10s}  {'Final':>10s}")
    print("-" * 70)
    for label, sv, sh, f in results:
        print(f"  {label:<26s}  {sv:>+10.2f}  {sh:>+10.2f}  ${f:>8,.0f}")
    print("=" * 70)

    # Veredito
    base_ho = results[0][2]
    t1_ho = results[1][2]
    t2_ho = results[2][2]
    print("\nVEREDITOS:")
    if abs(t1_ho) < 0.3:
        print(f"  ✓ T1 (shuffle): HOLDOUT Sharpe {t1_ho:+.2f} próximo de 0. Não há leak de label.")
    else:
        print(f"  ❌ T1 (shuffle): HOLDOUT Sharpe {t1_ho:+.2f} > |0.3|. POSSÍVEL LEAK.")
    diff = abs(base_ho - t2_ho)
    print(f"  • T2 (noise): base {base_ho:+.2f} vs com noise {t2_ho:+.2f}, delta {diff:.2f}.")
    print("    Variação grande indica que features reais carregam pouco signal vs ruído.")


if __name__ == "__main__":
    main()
