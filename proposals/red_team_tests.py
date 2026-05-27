"""Red-team adversarial tests para o pipeline btc-forecast.

Três testes diagnósticos para verificar se o Sharpe 1.29 é real ou artefato:

  1. shuffle_labels_test()    -> Sharpe deve colapsar a ~0. Se >0.3, há leak.
  2. time_reversed_test()     -> treina em futuro, testa em passado. Se Sharpe alto,
                                  modelo é não-causal (correlação estatica, não preditiva).
  3. noise_feature_test()     -> injeta feature N(0,1) iid. Se importance > 0.05,
                                  há leak no pipeline (normalização cross-fold etc).

Roda standalone:
    python proposals/red_team_tests.py
    python proposals/red_team_tests.py --test shuffle
    python proposals/red_team_tests.py --test reversed
    python proposals/red_team_tests.py --test noise

Notas:
  - Usa o mesmo timeframe/horizon/threshold do exp_ensemble.py para apples-to-apples.
  - Mantém custo COST=0.0008 do projeto (para isolar efeito do leak; aumentar para 0.002
    é um teste DIFERENTE, listado no brief).
  - Walk-forward expanding quarterly, purge=HORIZON_BARS.
"""
from __future__ import annotations

import argparse
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

# Mesmos hiperparâmetros do exp_ensemble.py / pipeline/model.py
TIMEFRAME = 240          # 4h
HORIZON_BARS = 12        # 48h, mid horizon
ATR_MULT = 3.0
COST = 0.0008
THRESHOLD = 0.35
BARS_PER_YEAR = 6 * 365  # 4h => 6 bars/day

PARAMS = dict(
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


# ============================================================================
# Helpers
# ============================================================================
def build_matrix() -> tuple[pd.DataFrame, list[str]]:
    """Replica exatamente o que exp_ensemble.py faz: feature matrix v2 + triple-barrier mid."""
    df = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME, lag=1).drop_nulls(subset=["atr_14"])
    labeled = lab.triple_barrier(df, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=HORIZON_BARS)
    labeled = labeled.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y"))
    fc = [
        c for c in labeled.columns
        if c not in feat.LAG_SAFE_EXCLUDE
        and c not in {"label", "hit_bar", "barrier_ret", "upper_px", "lower_px", "y"}
    ]
    mat = labeled.select(["open_time", "close", "y", "barrier_ret", *fc]).drop_nulls(
        subset=fc + ["y"]
    ).to_pandas()
    mat["dt"] = mat["open_time"].apply(lambda ms: datetime.fromtimestamp(ms / 1000, tz=timezone.utc))
    mat["quarter"] = mat["dt"].dt.to_period("Q")
    return mat, fc


def sharpe_per_trade(pnls: np.ndarray, n_years: float) -> float:
    """Sharpe anualizado por trade × √(trades/year). Espelha exp_ensemble.py."""
    if len(pnls) < 2 or pnls.std(ddof=1) == 0 or n_years <= 0:
        return 0.0
    tpy = len(pnls) / n_years
    return float(pnls.mean() / pnls.std(ddof=1) * np.sqrt(tpy))


def walk_forward(
    mat: pd.DataFrame,
    fc: list[str],
    y_col: str = "y",
    reversed_time: bool = False,
    verbose: bool = False,
) -> dict:
    """Walk-forward expanding quarterly. Acumula probabilities + retornos + dts."""
    if reversed_time:
        # Inverte ordem: o "futuro" vira treino, "passado" vira teste.
        # Mantemos quarters chronological mas iteramos em ordem reversa.
        # Para simplificar: ordenamos por DESCENDING e tratamos como timeline normal.
        # NB: index reset para que iloc seja válido após sort.
        mat = mat.sort_values("dt", ascending=False).reset_index(drop=True).copy()
        mat["quarter"] = mat["dt"].dt.to_period("Q")

    quarters = [q for q in sorted(mat["quarter"].unique()) if q.start_time.year >= 2023]
    all_proba, all_ret, all_dt = [], [], []

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
        train_idx = list(range(0, train_end))
        test_use_idx = [i for i in test_idx if i >= test_use_start]

        X_tr = mat.iloc[train_idx][fc].values
        y_tr = mat.iloc[train_idx][y_col].values
        X_te = mat.iloc[test_use_idx][fc].values
        ret_te = mat.iloc[test_use_idx]["barrier_ret"].values

        model = lgb.train(PARAMS, lgb.Dataset(X_tr, y_tr), num_boost_round=N_ROUNDS)
        proba = model.predict(X_te)
        all_proba.append(proba)
        all_ret.append(ret_te)
        all_dt.extend(mat.iloc[test_use_idx]["dt"].tolist())

        if verbose:
            print(f"  {q}  tr={len(train_idx):>5d}  te={len(test_use_idx):>4d}")

    if not all_proba:
        return {"sharpe": np.nan, "n_sig": 0, "tot_pnl": 0, "win_rate": 0}

    proba = np.concatenate(all_proba)
    ret = np.concatenate(all_ret)
    dts = pd.to_datetime(all_dt, utc=True)
    take = proba > THRESHOLD
    n_sig = int(take.sum())
    if n_sig < 5:
        return {"sharpe": 0.0, "n_sig": n_sig, "tot_pnl": 0.0, "win_rate": 0.0}

    pnls = ret[take] - COST
    sig_dts = dts[take]
    n_years = max(1e-9, (sig_dts.max() - sig_dts.min()).days / 365.25)
    sh = sharpe_per_trade(pnls, n_years)
    return {
        "sharpe": sh,
        "n_sig": n_sig,
        "tot_pnl": float(pnls.sum()),
        "win_rate": float((pnls > 0).mean()),
        "model_last": model,  # último modelo do walk-forward p/ noise test
    }


# ============================================================================
# Test 1 — Shuffle labels
# ============================================================================
def shuffle_labels_test(mat: pd.DataFrame, fc: list[str], n_runs: int = 3, seed: int = SEED) -> None:
    """Permuta y aleatoriamente. Sharpe deve colapsar a ~0.

    Se Sharpe shuffle > 0.3, há leak no pipeline (features veem o label de outra forma,
    ex.: target encoding, vazamento de futuro em normalização, etc).
    """
    print("\n" + "=" * 70)
    print("TEST 1 — SHUFFLE LABELS  (Sharpe esperado: 0 ± ruído)")
    print("=" * 70)

    rng = np.random.default_rng(seed)
    sharpes = []
    for run in range(n_runs):
        mat_sh = mat.copy()
        # Permuta y_global preservando posições; barrier_ret também precisa permutar JUNTO
        # (para que sinais que "acertam" reflitam um label aleatório, não retorno real).
        perm = rng.permutation(len(mat_sh))
        mat_sh["y"] = mat_sh["y"].values[perm]
        mat_sh["barrier_ret"] = mat_sh["barrier_ret"].values[perm]

        t0 = time.time()
        res = walk_forward(mat_sh, fc, y_col="y")
        elapsed = time.time() - t0
        sharpes.append(res["sharpe"])
        print(
            f"  run {run + 1}/{n_runs}  Sharpe={res['sharpe']:+.3f}  "
            f"n_sig={res['n_sig']:>4d}  win={100*res['win_rate']:.1f}%  "
            f"tot_pnl={100*res['tot_pnl']:+.1f}%  ({elapsed:.0f}s)"
        )

    mean_sh = float(np.mean(sharpes))
    print(f"\n  Média Sharpe shuffle: {mean_sh:+.3f}")
    if abs(mean_sh) > 0.30:
        print(f"  >>> FAIL: Sharpe shuffle {mean_sh:+.3f} > 0.30 — SUSPEITA DE LEAK no pipeline.")
    elif abs(mean_sh) > 0.15:
        print(f"  >>> WARN: Sharpe shuffle {mean_sh:+.3f} — possível leak residual ou variância grande.")
    else:
        print(f"  >>> PASS: Sharpe shuffle {mean_sh:+.3f} próximo de 0. Pipeline aparentemente sem leak grosso.")


# ============================================================================
# Test 2 — Time reversed
# ============================================================================
def time_reversed_test(mat: pd.DataFrame, fc: list[str]) -> None:
    """Treina em "futuro", testa em "passado". Walk-forward iterado em ordem reversa.

    Se Sharpe permanece alto (> 0.5), modelo está aprendendo correlação estática
    (ex.: mean-reversion universal, sazonalidade), não regime preditivo.
    Se Sharpe quebra para ≤ 0, pipeline é minimamente causal.
    """
    print("\n" + "=" * 70)
    print("TEST 2 — TIME-REVERSED  (Sharpe esperado: ≤ 0 se modelo é causal)")
    print("=" * 70)

    # Baseline (normal direction)
    print("\n  -- Forward (baseline, sanity check) --")
    t0 = time.time()
    res_fwd = walk_forward(mat, fc, y_col="y", reversed_time=False)
    print(
        f"  Sharpe_fwd={res_fwd['sharpe']:+.3f}  n_sig={res_fwd['n_sig']}  "
        f"win={100*res_fwd['win_rate']:.1f}%  tot={100*res_fwd['tot_pnl']:+.1f}%  "
        f"({time.time() - t0:.0f}s)"
    )

    # Reversed
    print("\n  -- Reversed (treino em futuro, teste em passado) --")
    t0 = time.time()
    res_rev = walk_forward(mat, fc, y_col="y", reversed_time=True)
    print(
        f"  Sharpe_rev={res_rev['sharpe']:+.3f}  n_sig={res_rev['n_sig']}  "
        f"win={100*res_rev['win_rate']:.1f}%  tot={100*res_rev['tot_pnl']:+.1f}%  "
        f"({time.time() - t0:.0f}s)"
    )

    delta = res_rev["sharpe"] - res_fwd["sharpe"]
    print(f"\n  Δ Sharpe (rev - fwd): {delta:+.3f}")
    if res_rev["sharpe"] > 0.5:
        print(
            f"  >>> FAIL: Sharpe reverso {res_rev['sharpe']:+.3f} > 0.5. "
            "Modelo aprende correlação não-causal (sazonalidade, mean-reversion estática)."
        )
    elif res_rev["sharpe"] > 0.0:
        print(
            f"  >>> WARN: Sharpe reverso {res_rev['sharpe']:+.3f} positivo. "
            "Provavelmente parte do sinal é não-causal — investigar features estáticas."
        )
    else:
        print(
            f"  >>> PASS: Sharpe reverso {res_rev['sharpe']:+.3f} ≤ 0. "
            "Modelo depende de ordem temporal (=mais causal que não-causal)."
        )


# ============================================================================
# Test 3 — Noise feature
# ============================================================================
def noise_feature_test(mat: pd.DataFrame, fc: list[str], seed: int = SEED) -> None:
    """Injeta feature `noise ~ N(0,1)` iid. Roda walk-forward. Mede importance da noise.

    Se gain(noise) / gain(top_feature) > 0.05 em qualquer fold, há leak no pipeline
    (ex.: features escalonadas usando dataset inteiro, target encoding, etc).
    Esperado: noise rank no fim da lista, gain ~0.
    """
    print("\n" + "=" * 70)
    print("TEST 3 — NOISE FEATURE  (importance esperada: ~0)")
    print("=" * 70)

    rng = np.random.default_rng(seed)
    mat_n = mat.copy()
    mat_n["noise"] = rng.standard_normal(len(mat_n))
    fc_n = fc + ["noise"]

    t0 = time.time()
    res = walk_forward(mat_n, fc_n, y_col="y")
    elapsed = time.time() - t0

    model = res.get("model_last")
    if model is None:
        print("  >>> ERROR: walk_forward não retornou modelo. Abortando teste.")
        return

    imp = pd.DataFrame({
        "feature": fc_n,
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)

    noise_row = imp[imp["feature"] == "noise"]
    if noise_row.empty:
        print("  >>> ERROR: noise feature não está nas importances. Bug no test.")
        return

    noise_rank = int(noise_row.index[0]) + 1
    noise_gain = float(noise_row["gain"].iloc[0])
    top_gain = float(imp.iloc[0]["gain"])
    ratio = noise_gain / top_gain if top_gain > 0 else 0.0

    print(f"\n  Walk-forward concluído em {elapsed:.0f}s (Sharpe={res['sharpe']:+.3f})")
    print(f"  noise rank: #{noise_rank} de {len(fc_n)}")
    print(f"  noise gain: {noise_gain:.1f}  (top gain: {top_gain:.1f}  ratio: {ratio:.3f})")

    print("\n  Top 10 por gain:")
    print(imp.head(10).to_string(index=False))
    print("\n  Bottom 5 por gain:")
    print(imp.tail(5).to_string(index=False))

    if ratio > 0.05:
        print(
            f"\n  >>> FAIL: noise gain ratio {ratio:.3f} > 0.05. "
            "Modelo está usando ruído puro como sinal — leak no pipeline."
        )
    elif noise_rank < len(fc_n) * 0.5:
        print(
            f"\n  >>> WARN: noise está no top {noise_rank}/{len(fc_n)} ({100*noise_rank/len(fc_n):.0f}%). "
            "Esperado seria estar no bottom 50%. Investigar."
        )
    else:
        print(
            f"\n  >>> PASS: noise no fim da lista (#{noise_rank}/{len(fc_n)}), ratio {ratio:.3f}. "
            "Pipeline não está reaproveitando ruído."
        )


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        choices=["all", "shuffle", "reversed", "noise"],
        default="all",
        help="Qual teste rodar (default: all).",
    )
    parser.add_argument("--n-runs", type=int, default=3, help="Repeats do shuffle test.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print("Building feature matrix (timeframe=4h, horizon=12, threshold=0.35) ...")
    t0 = time.time()
    mat, fc = build_matrix()
    print(f"  shape: {mat.shape}, features: {len(fc)}  ({time.time() - t0:.1f}s)")
    print(f"  range: {mat['dt'].min().date()} -> {mat['dt'].max().date()}")
    print(f"  base rate y=1: {100*mat['y'].mean():.1f}%")

    if args.test in ("all", "shuffle"):
        shuffle_labels_test(mat, fc, n_runs=args.n_runs, seed=args.seed)

    if args.test in ("all", "reversed"):
        time_reversed_test(mat, fc)

    if args.test in ("all", "noise"):
        noise_feature_test(mat, fc, seed=args.seed)

    print("\n" + "=" * 70)
    print("Done. Cross-check com brief: /briefs/red_team.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
