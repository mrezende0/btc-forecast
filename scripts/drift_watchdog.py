"""drift_watchdog — detecta data drift entre baseline rolling e janela atual.

PSI (Population Stability Index) por feature + KS-test.
Alerta no Telegram se max PSI > LIMIT.

Baseline: ROLLING — 12 meses anteriores à janela current (exclui current).
Current:  últimos LOOKBACK_DAYS dias (default 30 = 180 bars 4h).

Por que rolling: modelo retreina inline em cada predict (predict_dual_horizon
treina ensemble do zero). Baseline fixo em 2023-2024 detecta drift de regime
inevitável (bull 2026 ≠ 2023) e gera alertas falsos. Rolling captura drift
recente — o que o modelo NÃO viu nos últimos retreinos.

PSI buckets (LdP / risk industry padrão):
  < 0.10  → estável
  0.10-0.25 → mudança moderada (atenção)
  > 0.25  → drift significativo (alerta)
  > 0.50  → drift severo (CRÍTICO — risco de modelo degradado)

Saída:
- stdout: tabela top features por PSI desc
- data/drift_history.parquet: append rolling (date, feature, psi, ks_stat, ks_pvalue)
- Telegram: alerta se max PSI > PSI_CRITICAL

Idempotente: rerodar no mesmo dia substitui a entrada do dia.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from pipeline import features as feat, storage  # noqa: E402

# ---------- params
LOOKBACK_DAYS = 30
BASELINE_DAYS = 365     # janela do baseline rolling (12 meses antes do current)
BARS_4H_PER_DAY = 6
BARS_LOOKBACK = LOOKBACK_DAYS * BARS_4H_PER_DAY  # 180
BARS_BASELINE = BASELINE_DAYS * BARS_4H_PER_DAY  # ~2190

PSI_WARN = 0.10
PSI_ALERT = 0.25
PSI_CRITICAL = 0.50
N_BUCKETS = 10

HISTORY_PATH = ROOT / "data" / "drift_history.parquet"


# ---------- PSI
def compute_psi(baseline: np.ndarray, current: np.ndarray, n_buckets: int = N_BUCKETS) -> float:
    """PSI usando quantis do baseline como bordas.

    PSI = sum((p_curr - p_base) * log(p_curr / p_base))
    Aplica epsilon mínimo (1e-4) pra evitar log(0) / divisão por 0.
    """
    b = baseline[~np.isnan(baseline) & ~np.isinf(baseline)]
    c = current[~np.isnan(current) & ~np.isinf(current)]
    if len(b) < 100 or len(c) < 10:
        return np.nan
    # bordas pelos quantis do baseline
    qs = np.linspace(0, 1, n_buckets + 1)
    edges = np.quantile(b, qs)
    edges = np.unique(edges)
    if len(edges) < 3:  # feature quase constante
        return np.nan
    edges[0] = -np.inf
    edges[-1] = np.inf
    p_base = np.histogram(b, bins=edges)[0] / len(b)
    p_curr = np.histogram(c, bins=edges)[0] / len(c)
    eps = 1e-4
    p_base = np.clip(p_base, eps, None)
    p_curr = np.clip(p_curr, eps, None)
    return float(np.sum((p_curr - p_base) * np.log(p_curr / p_base)))


def compute_ks(baseline: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov 2-sample. Retorna (statistic, pvalue)."""
    from scipy.stats import ks_2samp
    b = baseline[~np.isnan(baseline) & ~np.isinf(baseline)]
    c = current[~np.isnan(current) & ~np.isinf(current)]
    if len(b) < 100 or len(c) < 10:
        return (np.nan, np.nan)
    res = ks_2samp(b, c)
    return (float(res.statistic), float(res.pvalue))


# ---------- main
def main():
    print(">>> drift_watchdog — build matriz de features…")
    df = feat.build_v2_from_parquets(timeframe_min=240, lag=1, asset="BTC")
    if df.is_empty():
        raise SystemExit("matriz vazia")

    # Particiona em baseline rolling (12m antes) vs current (últimos 30d)
    df_pd = df.to_pandas()
    df_pd["open_time"] = df_pd["open_time"].astype("int64")
    bar_ms = 4 * 3600 * 1000  # 4h em ms
    last_ts = df_pd["open_time"].iloc[-1]
    current_start = last_ts - BARS_LOOKBACK * bar_ms
    baseline_end = current_start - bar_ms
    baseline_start = baseline_end - BARS_BASELINE * bar_ms
    baseline_mask = (df_pd["open_time"] >= baseline_start) & (df_pd["open_time"] <= baseline_end)
    current_mask = df_pd["open_time"] >= current_start
    base = df_pd[baseline_mask]
    curr = df_pd[current_mask]
    print(f"  baseline: {len(base):,} bars (rolling {BASELINE_DAYS}d antes do current)")
    print(f"  current:  {len(curr):,} bars (últimos {LOOKBACK_DAYS}d)")

    if len(curr) < 30:
        raise SystemExit(f"current period muito curto: {len(curr)} bars. Aguarde mais dados.")

    # Features candidatas (exclui non-features e OHLCV bruto)
    feature_cols = [c for c in df_pd.columns
                    if c not in feat.LAG_SAFE_EXCLUDE
                    and df_pd[c].dtype in (np.float64, np.float32, np.int64, np.int32, np.int8)]

    print(f"  testando {len(feature_cols)} features…\n")

    results = []
    for col in feature_cols:
        b = base[col].to_numpy(dtype=float)
        c = curr[col].to_numpy(dtype=float)
        psi = compute_psi(b, c)
        ks_stat, ks_p = compute_ks(b, c)
        results.append({
            "feature": col,
            "psi": psi,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_p,
            "base_mean": float(np.nanmean(b)),
            "curr_mean": float(np.nanmean(c)),
            "base_std": float(np.nanstd(b)),
            "curr_std": float(np.nanstd(c)),
        })

    results_df = pl.DataFrame(results).sort("psi", descending=True, nulls_last=True)

    # ---------- print
    print("=" * 110)
    print(" DRIFT REPORT — Top 20 por PSI desc")
    print(f" PSI buckets: <{PSI_WARN} estável | {PSI_WARN}-{PSI_ALERT} moderado | >{PSI_ALERT} ALERTA | >{PSI_CRITICAL} CRÍTICO")
    print("=" * 110)
    header = f"{'feature':<28}{'PSI':>10}{'KS stat':>10}{'KS pval':>12}{'μ base':>14}{'μ curr':>14}{'flag':>10}"
    print(header)
    print("-" * 110)
    top = results_df.head(20).to_dicts()
    for r in top:
        psi_v = r["psi"]
        psi_is_nan = psi_v is None or (isinstance(psi_v, float) and np.isnan(psi_v))
        if psi_is_nan:
            flag = "n/d"
            psi_str = "    n/d"
        elif psi_v > PSI_CRITICAL:
            flag = "CRÍTICO"
            psi_str = f"{psi_v:>7.3f}"
        elif psi_v > PSI_ALERT:
            flag = "ALERTA"
            psi_str = f"{psi_v:>7.3f}"
        elif psi_v > PSI_WARN:
            flag = "moderado"
            psi_str = f"{psi_v:>7.3f}"
        else:
            flag = "ok"
            psi_str = f"{psi_v:>7.3f}"
        ks_stat = r["ks_stat"] if r["ks_stat"] is not None and not np.isnan(r["ks_stat"]) else 0.0
        ks_p = r["ks_pvalue"] if r["ks_pvalue"] is not None and not np.isnan(r["ks_pvalue"]) else 1.0
        print(
            f"{r['feature']:<28}{psi_str}   KS={ks_stat:.3f}  p={ks_p:.4f}  "
            f"μ {r['base_mean']:+.3f}→{r['curr_mean']:+.3f}  [{flag}]"
        )

    # Summary — filtra null E NaN (booleanos quase-constantes geram PSI=NaN, não são drift)
    valid_psi = results_df.filter(pl.col("psi").is_not_null() & pl.col("psi").is_not_nan())
    max_psi = valid_psi["psi"].max() or 0.0
    n_critical = valid_psi.filter(pl.col("psi") > PSI_CRITICAL).height
    n_alert = valid_psi.filter((pl.col("psi") > PSI_ALERT) & (pl.col("psi") <= PSI_CRITICAL)).height
    n_warn = valid_psi.filter((pl.col("psi") > PSI_WARN) & (pl.col("psi") <= PSI_ALERT)).height
    print("\n" + "=" * 110)
    print(f" SUMMARY:  max PSI = {max_psi:.3f}")
    print(f"   CRÍTICO (>{PSI_CRITICAL}): {n_critical} features")
    print(f"   ALERTA  (>{PSI_ALERT}):    {n_alert} features")
    print(f"   moderado(>{PSI_WARN}):    {n_warn} features")
    print("=" * 110)

    # ---------- persist
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    snapshot = results_df.with_columns(
        pl.lit(today).alias("date"),
        pl.lit(int(datetime.now(tz=timezone.utc).timestamp() * 1000)).alias("generated_at_ms"),
    ).select(["date", "generated_at_ms", "feature", "psi", "ks_stat", "ks_pvalue",
              "base_mean", "curr_mean", "base_std", "curr_std"])

    if HISTORY_PATH.exists():
        prev = pl.read_parquet(HISTORY_PATH)
        # remove entradas do mesmo dia (idempotente)
        prev = prev.filter(pl.col("date") != today)
        combined = pl.concat([prev, snapshot], how="vertical_relaxed")
    else:
        combined = snapshot
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(HISTORY_PATH)
    print(f"\n>>> history salvo em {HISTORY_PATH} ({combined.height:,} rows acumulados)")

    # ---------- Telegram alert (usa só PSI válido — exclui NaN)
    critical_features = valid_psi.filter(pl.col("psi") > PSI_CRITICAL).head(5).to_dicts()
    alert_features = valid_psi.filter(
        (pl.col("psi") > PSI_ALERT) & (pl.col("psi") <= PSI_CRITICAL)
    ).head(5).to_dicts()

    if max_psi > PSI_ALERT and os.environ.get("TELEGRAM_BOT_TOKEN"):
        try:
            from pipeline import telegram as tg
            lines = ["🚨 *DRIFT DETECTADO* — modelo pode estar degradado",
                     f"`{today}` · baseline VAL vs últimos {LOOKBACK_DAYS}d",
                     ""]
            if critical_features:
                lines.append(f"🔴 *CRÍTICO* (PSI > {PSI_CRITICAL}):")
                for r in critical_features:
                    lines.append(f"  • `{r['feature']}` PSI={r['psi']:.2f}  "
                                 f"μ {r['base_mean']:+.3f}→{r['curr_mean']:+.3f}")
            if alert_features:
                lines.append("")
                lines.append(f"🟡 *ALERTA* (PSI > {PSI_ALERT}):")
                for r in alert_features:
                    lines.append(f"  • `{r['feature']}` PSI={r['psi']:.2f}")
            lines.append("")
            lines.append(f"_max PSI = {max_psi:.2f} · {n_critical} críticos · {n_alert} alertas · {n_warn} moderados_")
            msg = "\n".join(lines)
            tg.send(msg)
            print(">>> Telegram alert enviado")
        except Exception as e:
            print(f">>> Telegram alert FALHOU: {e}")
    elif max_psi > PSI_ALERT:
        print(">>> drift detectado mas TELEGRAM_BOT_TOKEN ausente — sem alerta")
    else:
        print(">>> nenhum drift crítico — sem alerta")


if __name__ == "__main__":
    main()
