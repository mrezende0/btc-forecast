# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # cpcv_dsr_audit — auditoria estatística do Sharpe 1.29
#
# Stub executável (em partes — funcoes implementadas, integracao end-to-end TODO).
#
# Objetivos:
#   1. Carregar os resultados dos exp_* já rodados (research log + Sharpes observados).
#   2. Computar Probabilistic Sharpe Ratio (PSR) e Deflated Sharpe Ratio (DSR)
#      dado K_trials honesto.
#   3. IC 95% para Sharpe via stationary bootstrap (Politis-Romano 1994).
#   4. Esqueleto de CPCV usando mlfinpy (gera distribuicao de Sharpes em N paths).
#   5. PBO via CSCV (Bailey-Borwein-Lopez de Prado-Zhu 2017).
#
# Referencias:
#   - Bailey & Lopez de Prado (2014), Deflated Sharpe Ratio.
#   - Bailey & Lopez de Prado (2012), Probabilistic Sharpe Ratio.
#   - Lopez de Prado (2018), AFML caps. 7, 11, 14.
#   - Politis & Romano (1994), Stationary Bootstrap.
#   - Harvey & Liu (2015), Backtesting (haircut).

# %%
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Constantes do projeto
TIMEFRAME_MIN = 240
HORIZON_MID = 12
HORIZON_LONG = 18
COST = 0.0008
EULER_MASCHERONI = 0.5772156649

# %% [markdown]
# ## 1. Research log — K_trials honesto
#
# Lista (manual, pre-registrada) dos experimentos rodados ate aqui.
# Cada linha = 1 trial. Source: notebooks/exp_*.py + 07_hyperopt.py.
# OBS: subestima K real (cada Optuna trial conta como N trials).

# %%
RESEARCH_LOG: list[dict] = [
    # 07_hyperopt.py — 30 trials Optuna otimizando Sharpe direto na VAL
    {"exp": "07_hyperopt", "n_configs": 30, "notes": "Optuna TPE 30 trials, otimiza Sharpe VAL"},
    # exp_threshold_grid — 25 combos mid_thr x long_thr
    {"exp": "exp_threshold_grid", "n_configs": 25, "notes": "5x5 grid (mid_thr, long_thr)"},
    # exp_ensemble — 5 regras de combinacao
    {"exp": "exp_ensemble", "n_configs": 5, "notes": "LGB/XGB/mean/weighted/max"},
    # exp_multi_horizon — 3 horizontes standalone + 3 regras voto
    {"exp": "exp_multi_horizon", "n_configs": 6, "notes": "short/mid/long + consensus/majority/any"},
    # outros experimentos cada um conta como pelo menos 3 variacoes
    {"exp": "exp_position_sizing", "n_configs": 5, "notes": "estimado, varias regras de sizing"},
    {"exp": "exp_regime_analysis", "n_configs": 4, "notes": "estimado"},
    {"exp": "exp_drawdown_analysis", "n_configs": 1, "notes": "diagnostico, nao selecao"},
    {"exp": "exp_ema200_veto", "n_configs": 3, "notes": "estimado"},
    {"exp": "exp_wick_filter", "n_configs": 3, "notes": "estimado"},
    {"exp": "exp_asym_barriers", "n_configs": 4, "notes": "estimado"},
    {"exp": "exp_backtest_1k", "n_configs": 1, "notes": "diagnostico"},
    # baselines/features/labels iteration — minimo 5 redesenhos
    {"exp": "feature_engineering", "n_configs": 5, "notes": "v1->v2 + redesigns documentados em commits"},
]


def k_trials_total(log: Iterable[dict]) -> int:
    return int(sum(x["n_configs"] for x in log))


# %% [markdown]
# ## 2. Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012)
#
# PSR(SR*) = Phi( (SR_hat - SR*) * sqrt(n - 1) / sqrt(1 - g3*SR_hat + (g4-1)/4 * SR_hat^2) )
#
# - SR_hat: Sharpe observado (nao anualizado — por periodo de retorno).
# - SR*: threshold (zero para "ha edge?"; SR=alvo/anualizar_fator para hipotese forte).
# - g3, g4: skewness e kurtose dos retornos por trade.
# - n: numero de observacoes (trades).

# %%
def psr(sr_hat: float, sr_star: float, returns: np.ndarray) -> float:
    """Probabilistic Sharpe Ratio. sr_hat e sr_star em mesma escala (por trade).

    Retorna probabilidade que o true SR seja > sr_star.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    if n < 3:
        return float("nan")
    g3 = float(stats.skew(returns, bias=False))
    g4 = float(stats.kurtosis(returns, fisher=False, bias=False))  # nao-excess
    denom = np.sqrt(1.0 - g3 * sr_hat + ((g4 - 1.0) / 4.0) * sr_hat ** 2)
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    z = (sr_hat - sr_star) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


# %% [markdown]
# ## 3. Expected Maximum Sharpe sob nula (Bailey & Lopez de Prado 2014)
#
# SR_0 = sqrt(V[SR_hat]) * ((1 - gamma_em) * Phi^-1(1 - 1/N)
#                          + gamma_em * Phi^-1(1 - 1/(N*e)))

# %%
def expected_max_sr(var_sr: float, n_trials: int) -> float:
    """Expected max Sharpe ratio sob H0 (true SR=0) dado N trials independentes."""
    if n_trials < 2 or var_sr <= 0:
        return 0.0
    g = EULER_MASCHERONI
    e = np.e
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
    return float(np.sqrt(var_sr) * ((1 - g) * z1 + g * z2))


def dsr(sr_hat: float, returns: np.ndarray, sharpes_observed: Sequence[float], n_trials: int) -> float:
    """Deflated Sharpe Ratio = PSR(SR_0) usando expected_max_sr como threshold.

    sharpes_observed: lista de Sharpes (mesma escala) dos N trials rodados —
    usado pra estimar V[SR_hat] entre experimentos.
    """
    if len(sharpes_observed) < 2:
        var_sr = float("nan")
    else:
        var_sr = float(np.var(sharpes_observed, ddof=1))
    sr_0 = expected_max_sr(var_sr, n_trials)
    return psr(sr_hat, sr_0, returns)


# %% [markdown]
# ## 4. Stationary bootstrap IC pro Sharpe (Politis-Romano 1994)
#
# Block size otimo ~ sqrt(n). Para evitar dependencia de lib arch, implementacao
# manual (geometric block lengths).

# %%
def stationary_bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_boot: int = 5000,
    block_size: float | None = None,
    confidence: float = 0.95,
    annualize_factor: float = 1.0,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Retorna (sharpe_point, ci_lower, ci_upper) anualizado por annualize_factor."""
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < 5:
        return (float("nan"), float("nan"), float("nan"))
    if block_size is None:
        block_size = float(np.sqrt(n))
    p = 1.0 / block_size  # prob de comecar novo bloco

    def sharpe(x: np.ndarray) -> float:
        if x.std(ddof=1) == 0:
            return 0.0
        return float(x.mean() / x.std(ddof=1) * annualize_factor)

    sr_hat = sharpe(r)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        i = int(rng.integers(0, n))
        for t in range(n):
            idx[t] = i % n
            if rng.random() < p:
                i = int(rng.integers(0, n))
            else:
                i += 1
        boots[b] = sharpe(r[idx])
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1.0 - alpha))
    return sr_hat, lo, hi


# %% [markdown]
# ## 5. Esqueleto CPCV via mlfinpy
#
# CPCV(N=10, k=2) -> C(10,2)=45 splits -> phi[10,2]=9 backtest paths.
# Cada path gera uma equity curve OOS completa. Sharpe por path -> distribuicao.
#
# Implementacao real depende de:
#   - mlfinpy.cross_validation.combinatorial.CombinatorialPurgedCV
#   - eventos com t1 (label endtime) para purge correto
#
# Stub abaixo descreve a integracao; chamada real precisa que
# `pipeline/labels.py` exponha t1 (que ja tem em hit_bar).

# %%
def cpcv_paths_sharpe(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    returns: pd.Series,
    model_fn: Callable[[pd.DataFrame, pd.Series], object],
    predict_fn: Callable[[object, pd.DataFrame], np.ndarray],
    threshold: float = 0.35,
    n_splits: int = 10,
    n_test_splits: int = 2,
    embargo_pct: float = 0.01,
    cost: float = COST,
) -> dict:
    """Roda CPCV e retorna distribuicao de Sharpe por path.

    Dependencia: pip install mlfinpy (ja no projeto).
    """
    try:
        from mlfinpy.cross_validation.combinatorial import (
            CombinatorialPurgedKFold,
        )
    except Exception as e:
        raise ImportError(
            "mlfinpy CombinatorialPurgedKFold ausente. "
            "Verifique versao/instalacao: pip install -U mlfinpy"
        ) from e

    cv = CombinatorialPurgedKFold(
        n_splits=n_splits,
        n_test_splits=n_test_splits,
        samples_info_sets=t1,
        pct_embargo=embargo_pct,
    )

    # Cada combinacao (C(n_splits, n_test_splits)) gera 1 split.
    # CPCV recombina pra phi[N,k] paths -> aqui simplificamos:
    # rodamos por SPLIT, depois agrupamos predicoes por path via mlfinpy helper.
    path_pnls: dict[int, list[float]] = {}
    # mlfinpy expoe get_backtest_paths(); se nao, agrupar manualmente.
    for split_id, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
        model = model_fn(X.iloc[tr_idx], y.iloc[tr_idx])
        proba = predict_fn(model, X.iloc[te_idx])
        take = proba > threshold
        pnls = (returns.iloc[te_idx].values[take] - cost).tolist()
        # mapeia split_id -> path_id (mlfinpy.cv.get_path_assignment)
        # TODO: usar API real; aqui placeholder split_id % phi[N,k]
        n_paths = _phi(n_splits, n_test_splits)
        path_id = split_id % n_paths
        path_pnls.setdefault(path_id, []).extend(pnls)

    out: dict = {}
    for pid, pnls in path_pnls.items():
        arr = np.array(pnls)
        if len(arr) < 5 or arr.std(ddof=1) == 0:
            out[pid] = {"n": len(arr), "sharpe": float("nan")}
            continue
        n_years = max(1e-9, len(arr) / (6 * 365))  # 6 bars/dia em 4h
        sh = arr.mean() / arr.std(ddof=1) * np.sqrt(len(arr) / n_years)
        out[pid] = {"n": len(arr), "sharpe": float(sh), "tot": float(arr.sum())}
    return out


def _phi(n: int, k: int) -> int:
    """Numero de backtest paths CPCV (Lopez de Prado 2018, AFML eq. 12.1)."""
    from math import comb
    return comb(n - 1, k - 1) if n > 0 and k > 0 else 0


# %% [markdown]
# ## 6. PBO via CSCV (Bailey-Borwein-Lopez de Prado-Zhu 2017)
#
# Input: matriz M (T x N) = retornos por periodo (T) x configuracao testada (N).
# Para cada combinacao de S/2 sub-grupos como IS / S/2 como OOS (S=16 default):
#   1. acha config #1 em IS (max Sharpe IS).
#   2. mede rank OOS dessa mesma config.
#   3. computa logit lambda = log(rank/(N+1-rank)).
# PBO = P(lambda <= 0) = P(melhor IS fica na metade inferior OOS).

# %%
def pbo_cscv(returns_matrix: np.ndarray, n_subgroups: int = 16) -> dict:
    """Probability of Backtest Overfitting via CSCV.

    returns_matrix: shape (T, N_configs) — retornos por periodo, por configuracao.
    """
    from itertools import combinations

    T, N = returns_matrix.shape
    S = n_subgroups
    if T < S * 2:
        raise ValueError(f"T={T} insuficiente para S={S}. Reduza n_subgroups.")
    # parte T em S sub-grupos contiguos (preserva ordem temporal)
    group_size = T // S
    groups = [returns_matrix[i * group_size:(i + 1) * group_size] for i in range(S)]

    lambdas: list[float] = []
    half = S // 2
    for is_combo in combinations(range(S), half):
        oos_combo = tuple(g for g in range(S) if g not in is_combo)
        is_block = np.vstack([groups[g] for g in is_combo])
        oos_block = np.vstack([groups[g] for g in oos_combo])

        # Sharpe por config em IS e OOS (assume mean/std por periodo)
        def sr_per_config(M: np.ndarray) -> np.ndarray:
            mu = M.mean(axis=0)
            sd = M.std(axis=0, ddof=1)
            sd = np.where(sd == 0, np.nan, sd)
            return mu / sd

        sr_is = sr_per_config(is_block)
        sr_oos = sr_per_config(oos_block)
        if np.all(np.isnan(sr_is)):
            continue
        best_is = int(np.nanargmax(sr_is))
        # rank OOS da best_is (1=pior, N=melhor)
        valid = ~np.isnan(sr_oos)
        ranks = stats.rankdata(np.where(valid, sr_oos, -np.inf), method="average")
        rank_best = ranks[best_is]
        omega = rank_best / (N + 1)
        lam = float(np.log(omega / (1.0 - omega))) if 0 < omega < 1 else float("nan")
        if np.isfinite(lam):
            lambdas.append(lam)

    lambdas_arr = np.array(lambdas)
    pbo = float((lambdas_arr <= 0).mean()) if len(lambdas_arr) else float("nan")
    return {"pbo": pbo, "n_combos": len(lambdas_arr), "lambdas": lambdas_arr}


# %% [markdown]
# ## 7. Haircut Harvey-Liu 2015 (Bonferroni simples como sanity)
#
# t-stat adjusted = t-stat_observed / sqrt(K).
# Sharpe haircut ~ proportional to t-stat ratio.

# %%
def harvey_liu_haircut(sr_hat: float, n_obs: int, n_trials: int, freq: float = 1.0) -> dict:
    """Haircut Bonferroni simples (Harvey-Liu apresentam Holm e BHY tambem;
    este e o caso mais conservador)."""
    # t-stat associada ao Sharpe anualizado (assume freq=trades/ano para nao-anualizar)
    sr_per_period = sr_hat / np.sqrt(freq) if freq > 1 else sr_hat
    t_obs = sr_per_period * np.sqrt(n_obs)
    p_single = 2 * (1 - stats.norm.cdf(abs(t_obs)))
    p_adj = min(1.0, p_single * n_trials)  # Bonferroni
    # haircut: p_adj -> t_adj -> sr_adj
    if p_adj >= 1.0:
        sr_adj = 0.0
    else:
        t_adj = stats.norm.ppf(1 - p_adj / 2)
        sr_adj_per_period = t_adj / np.sqrt(n_obs)
        sr_adj = sr_adj_per_period * np.sqrt(freq) if freq > 1 else sr_adj_per_period
    return {
        "sr_observed": sr_hat,
        "sr_adjusted_bonferroni": sr_adj,
        "haircut_pct": 100.0 * (1.0 - sr_adj / sr_hat) if sr_hat > 0 else 0.0,
        "p_single": p_single,
        "p_adjusted": p_adj,
        "n_trials": n_trials,
    }


# %% [markdown]
# ## 8. Carrega returns reais do projeto e roda auditoria
#
# TODO: implementar `load_dual_horizon_trades()` lendo o pipeline real
# (replicar `exp_drawdown_analysis.py` mas retornando os PnLs por trade
# em vez de printar).
#
# Esqueleto:

# %%
def load_dual_horizon_trades() -> pd.DataFrame:
    """Replica producao: walk-forward 2023+, mid AND long > 0.35.
    Retorna DF com colunas: open_time, pnl_net (pos custo).

    TODO: refatorar exp_drawdown_analysis pra expor essa funcao.
    Por ora, le um CSV pre-gerado se existir.
    """
    cache = ROOT / "notebooks" / "dual_horizon_trades.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["open_time"])
    raise FileNotFoundError(
        f"{cache} ausente. Rodar primeiro: refatorar exp_drawdown_analysis "
        "para salvar os PnLs por trade em CSV antes de auditar."
    )


# %% [markdown]
# ## 9. Pipeline end-to-end (rode tudo)

# %%
def run_audit(sharpe_anualizado_reportado: float = 1.29) -> dict:
    """Executa todos os testes e imprime relatorio.

    Returns dict com todos os numeros, util pra serializar em JSON.
    """
    print("=" * 78)
    print("AUDITORIA — Dual-horizon AND ensemble (Sharpe reportado = 1.29)")
    print("=" * 78)

    k = k_trials_total(RESEARCH_LOG)
    print(f"\n[1] K_trials honesto (research log): {k}")
    print(f"    NB: subestima K real — cada trial Optuna conta como 1.")

    trades = load_dual_horizon_trades()
    pnls = trades["pnl_net"].values
    n = len(pnls)
    print(f"\n[2] Trades carregados: n={n}")

    # Sharpe por trade (nao anualizado) e fator de anualizacao
    if pnls.std(ddof=1) == 0:
        raise RuntimeError("std=0 — verifique dados.")
    sr_per_trade = pnls.mean() / pnls.std(ddof=1)
    # n_anos do pool — usar timestamps reais
    n_anos = (trades["open_time"].max() - trades["open_time"].min()).days / 365.25
    trades_per_year = n / max(n_anos, 1e-9)
    annualize = np.sqrt(trades_per_year)
    sr_annual = sr_per_trade * annualize
    print(f"    SR por trade: {sr_per_trade:.4f}  | annualize_factor: {annualize:.2f}")
    print(f"    SR anualizado recomputado: {sr_annual:.3f}  (reportado: {sharpe_anualizado_reportado})")

    # IC bootstrap
    sr_pt, lo, hi = stationary_bootstrap_sharpe_ci(
        pnls, n_boot=5000, annualize_factor=annualize
    )
    print(f"\n[3] IC95% stationary bootstrap: [{lo:+.3f}, {hi:+.3f}]")
    print(f"    Sharpe sobrevive ao IC se lo > 0.5: {'SIM' if lo > 0.5 else 'NAO'}")

    # PSR
    psr_zero = psr(sr_per_trade, 0.0, pnls)
    psr_alvo = psr(sr_per_trade, 1.0 / annualize, pnls)  # SR anualizado=1 em escala/trade
    print(f"\n[4] PSR(SR*=0):   {psr_zero:.3f}   (criterio >0.95)")
    print(f"    PSR(SR*=1.0): {psr_alvo:.3f}   (bonus  >0.50)")

    # DSR — V[SR] precisa de lista de Sharpes obs entre experimentos.
    # TODO: extrair Sharpes dos exp_*_results.csv quando existirem.
    sharpes_observed = [0.50, 0.88, 1.29, 0.72, 0.91, 1.05, 0.66, 0.40, 1.10, 0.55]  # placeholder
    dsr_val = dsr(sr_per_trade, pnls, sharpes_observed, n_trials=k)
    sr_0 = expected_max_sr(np.var(sharpes_observed, ddof=1), k)
    print(f"\n[5] Expected max SR sob H0 (K={k}): {sr_0:.4f} per-trade  "
          f"({sr_0*annualize:.3f} anualizado)")
    print(f"    DSR = PSR(SR_0): {dsr_val:.3f}   (criterio >0.95)")

    # Haircut Harvey-Liu
    hl = harvey_liu_haircut(sr_annual, n, k, freq=trades_per_year)
    print(f"\n[6] Harvey-Liu haircut (Bonferroni):")
    print(f"    SR_obs:     {hl['sr_observed']:+.3f}")
    print(f"    SR_adj:     {hl['sr_adjusted_bonferroni']:+.3f}")
    print(f"    haircut:    {hl['haircut_pct']:.1f}%")
    print(f"    p_single:   {hl['p_single']:.4f}")
    print(f"    p_adjusted: {hl['p_adjusted']:.4f}")

    print("\n[7] CPCV — TODO: integrar com pipeline.model.fit / mlfinpy.")
    print("    Esqueleto em `cpcv_paths_sharpe()`.")
    print("\n[8] PBO/CSCV — TODO: gerar returns_matrix (T, N_configs).")
    print("    Esqueleto em `pbo_cscv()`.")

    veredito = (
        "REPROVADO" if (lo < 0.5 or psr_zero < 0.95 or dsr_val < 0.95)
        else "APROVADO sob criterio rigoroso"
    )
    print("\n" + "=" * 78)
    print(f"VEREDITO: {veredito}")
    print("=" * 78)

    return {
        "k_trials": k,
        "n_trades": n,
        "sharpe_annual": sr_annual,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "psr_zero": psr_zero,
        "psr_alvo": psr_alvo,
        "expected_max_sr": sr_0 * annualize,
        "dsr": dsr_val,
        "haircut_pct": hl["haircut_pct"],
        "sr_haircut": hl["sr_adjusted_bonferroni"],
        "veredito": veredito,
    }


# %%
if __name__ == "__main__":
    try:
        result = run_audit(sharpe_anualizado_reportado=1.29)
    except FileNotFoundError as e:
        print(f"[setup] {e}")
        print("\nProximo passo: refatorar exp_drawdown_analysis.py para expor")
        print("`load_dual_horizon_trades()` salvando notebooks/dual_horizon_trades.csv.")
