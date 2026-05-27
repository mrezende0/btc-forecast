"""exp_accuracy_report — Análise detalhada de acurácia do modelo no histórico.

Diferente dos backtests focados em Sharpe/PnL, este foca em CALIBRAÇÃO:
- Predicted proba vs actual outcome (por bin)
- Confusion matrix
- Win rate por mês (consistência temporal)
- Brier score (qualidade da probabilidade)
- Sinal vs ruído por janela de proba

Usa probas cached do A1-A (data/walk_forward_probas.parquet).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

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
THR = 0.35
NO_BEAR = -0.05
BARS_PER_MONTH = 180


def build_matrix():
    df = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME, lag=1).drop_nulls(subset=["atr_14"])
    lab_df = lab.triple_barrier(df, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=HORIZON)
    lab_df = lab_df.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y"))
    m = lab_df.select(["open_time", "close", "high", "low", "atr_14", "y", "barrier_ret"]).drop_nulls().to_pandas()
    m["dt"] = m["open_time"].apply(lambda ms: datetime.fromtimestamp(ms/1000, tz=timezone.utc))
    return m


# Carrega probas cached
probas_path = ROOT / "data" / "walk_forward_probas.parquet"
if not probas_path.exists():
    raise FileNotFoundError(f"{probas_path} não existe. Rode exp_a1_threshold_search primeiro.")

probas_df = pd.read_parquet(probas_path)
print(f"Probas cached: {len(probas_df)} rows, cols={list(probas_df.columns)}")

mat = build_matrix()
print(f"Matrix: {len(mat)} rows")

# Merge probas com labels via open_time
merged = mat.merge(probas_df[["open_time", "proba_mid"]], on="open_time", how="inner")
print(f"Merged: {len(merged)} rows com proba + label")

# Filtra só barras cobertas (com proba > 0)
merged = merged[merged["proba_mid"] > 0].reset_index(drop=True)
print(f"Cobertas (proba>0): {len(merged)} rows")
print(f"Range: {merged['dt'].min()} → {merged['dt'].max()}\n")

# ============================================================
# 1. CALIBRAÇÃO — proba prevista vs win rate observado
# ============================================================
print("=" * 80)
print(" 1. CALIBRAÇÃO — probabilidade prevista vs taxa de acerto observada")
print("=" * 80)

bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.35), (0.35, 0.4),
        (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]
print(f"\n{'proba bin':<18s}  {'n':>6s}  {'mean proba':>11s}  {'win rate real':>14s}  {'gap':>6s}")
print("-" * 70)
for lo, hi in bins:
    mask = (merged["proba_mid"] >= lo) & (merged["proba_mid"] < hi)
    n = mask.sum()
    if n < 10:
        continue
    mp = merged.loc[mask, "proba_mid"].mean()
    wr = merged.loc[mask, "y"].mean()
    gap = wr - mp
    flag = "  ✓" if abs(gap) < 0.05 else ("  ⚠️" if abs(gap) < 0.10 else "  ❌")
    print(f"  [{lo:.2f}, {hi:.2f}]    {n:>6d}  {100*mp:>9.1f}%   {100*wr:>13.1f}%  {100*gap:>+5.1f}%{flag}")

print("\nLeitura: modelo BEM calibrado se win rate real ≈ proba prevista.")
print("Bins [0.35, 0.50] são os mais relevantes (zona de decisão).")

# ============================================================
# 2. CONFUSION MATRIX no threshold 0.35 (sem filtro bear)
# ============================================================
print("\n" + "=" * 80)
print(" 2. CONFUSION MATRIX @ threshold 0.35")
print("=" * 80)

merged["pred"] = (merged["proba_mid"] > THR).astype(int)
tp = ((merged["pred"] == 1) & (merged["y"] == 1)).sum()
fp = ((merged["pred"] == 1) & (merged["y"] == 0)).sum()
tn = ((merged["pred"] == 0) & (merged["y"] == 0)).sum()
fn = ((merged["pred"] == 0) & (merged["y"] == 1)).sum()

print(f"\n                  Pred LONG_WIN   Pred OUTRO")
print(f"  Real LONG_WIN      TP={tp:>5d}        FN={fn:>5d}")
print(f"  Real OUTRO         FP={fp:>5d}        TN={tn:>5d}")
print()
prec = tp / max(1, tp + fp)
rec = tp / max(1, tp + fn)
print(f"  Precision (acerto entre os sinais):  {100*prec:.1f}%")
print(f"  Recall (peg os LONG_WIN reais):      {100*rec:.1f}%")
print(f"  Base rate LONG_WIN:                  {100*merged['y'].mean():.1f}%")
print(f"  Edge precision: {100*(prec - merged['y'].mean()):.1f}pp acima do random")

# ============================================================
# 3. WIN RATE POR MÊS — consistência temporal
# ============================================================
print("\n" + "=" * 80)
print(" 3. WIN RATE POR MÊS dos sinais (proba > 0.35)")
print("=" * 80)

signals = merged[merged["proba_mid"] > THR].copy()
signals["month"] = signals["dt"].dt.to_period("M")

monthly = signals.groupby("month").agg(
    n=("y", "size"),
    win=("y", "mean"),
    avg_ret=("barrier_ret", "mean"),
).reset_index()

print(f"\n{'mês':<10s}  {'n sinais':>9s}  {'win%':>6s}  {'avg ret':>8s}")
print("-" * 45)
for _, row in monthly.iterrows():
    n = int(row['n'])
    if n < 1:
        continue
    print(f"  {str(row['month']):<10s}  {n:>9d}  {100*row['win']:>5.1f}%  {100*row['avg_ret']:>+6.2f}%")

print(f"\nMédia mensal win rate: {100*monthly['win'].mean():.1f}%")
print(f"Desvio padrão entre meses: {100*monthly['win'].std():.1f}pp")
print(f"Meses ≥ 50% win: {(monthly['win'] >= 0.5).sum()} de {len(monthly)}")

# ============================================================
# 4. BRIER SCORE — qualidade da probabilidade total
# ============================================================
print("\n" + "=" * 80)
print(" 4. BRIER SCORE — qualidade probabilística geral")
print("=" * 80)

brier = ((merged["proba_mid"] - merged["y"]) ** 2).mean()
base_rate = merged["y"].mean()
brier_base = ((base_rate - merged["y"]) ** 2).mean()  # baseline: prever sempre base rate
brier_random = 0.25  # baseline aleatório p=0.5

print(f"  Brier score modelo:       {brier:.4f}")
print(f"  Brier score base rate:    {brier_base:.4f}  (prever sempre {100*base_rate:.0f}%)")
print(f"  Brier score random p=0.5: {brier_random:.4f}")
print()
print(f"  ↓ menor é melhor. Modelo está {'MELHOR' if brier < brier_base else 'PIOR'} que base rate.")
print(f"  Skill (0=perfeito, 1=ruim): {brier / brier_base:.3f}")

# ============================================================
# 5. SINAL POR FAIXA — PnL e qualidade
# ============================================================
print("\n" + "=" * 80)
print(" 5. PnL por faixa de proba — onde modelo é forte?")
print("=" * 80)

ranges = [(0.35, 0.40), (0.40, 0.45), (0.45, 0.50), (0.50, 0.60), (0.60, 1.0)]
print(f"\n{'faixa proba':<18s}  {'n':>5s}  {'win%':>6s}  {'avg ret':>8s}  {'sharpe':>7s}")
print("-" * 60)
for lo, hi in ranges:
    mask = (merged["proba_mid"] >= lo) & (merged["proba_mid"] < hi)
    n = mask.sum()
    if n < 5:
        continue
    win = merged.loc[mask, "y"].mean()
    ar = merged.loc[mask, "barrier_ret"].mean() - COST
    sd = merged.loc[mask, "barrier_ret"].std()
    sr = ar / sd if sd > 0 else 0
    print(f"  [{lo:.2f}, {hi:.2f}]      {n:>5d}  {100*win:>5.1f}%  {100*ar:>+6.2f}%  {sr:>+6.2f}")

print("\n" + "=" * 80)
print(" CONCLUSÃO")
print("=" * 80)
print("""
Pra olhar:
- Calibração: bins [0.35, 0.50] devem ter win rate REAL próximo da proba média
  Se descalibrado, threshold 0.35 não significa "35% chance real"
- Precision @ thr 0.35: se > base rate por 10pp+ = edge real
- Win rate mensal: variância indica quanto a performance depende de regime
- Brier: skill < 1.0 = melhor que prever base rate sempre
""")
