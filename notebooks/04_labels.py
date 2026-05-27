# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # 04 — Triple-barrier labels
#
# Aplica triple-barrier no OHLCV+ATR já com features. Olha:
#   - distribuição das 3 classes (long_win / timeout / stop)
#   - tempo médio até barreira (deve ser bem menor que horizonte)
#   - validação de simetria upper/lower
#   - sanity de retornos por classe

# %%
from __future__ import annotations
from pathlib import Path
import sys, os
ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import polars as pl
import numpy as np
from pipeline import features as feat, labels as lab

# %% [markdown]
# ## 1. Build features (precisamos do ATR)

# %%
df = feat.build_from_parquets(lag=1)
# remove warm-up (todas features = NaN)
df = df.drop_nulls(subset=["atr_14"])
print(f"input shape: {df.shape}")

# %% [markdown]
# ## 2. Aplica triple-barrier
#
# Parâmetros default: ±1.5 × ATR, horizonte 32 bars (8h).

# %%
# Parâmetros calibrados em sweep (ver mensagem do chat).
# 3.0×ATR / 32 bars (8h) dá:
#   - ~29% timeout (sweet spot 20-50%)
#   - ±1.67% retorno por hit (>> custo 0.08%)
#   - simetria 0.94 (neutro)
UPPER = LOWER = 3.0
HORIZON_BARS = 32  # 8h
labeled = lab.triple_barrier(df, upper_mult=UPPER, lower_mult=LOWER, horizon_bars=HORIZON_BARS)
print(f"labeled shape: {labeled.shape}")

# %% [markdown]
# ## 3. Distribuição das classes

# %%
dist = labeled.group_by("label").agg(
    pl.len().alias("count"),
    pl.col("hit_bar").mean().alias("hit_bar_mean"),
    pl.col("hit_bar").median().alias("hit_bar_median"),
    pl.col("barrier_ret").mean().alias("ret_mean"),
).sort("label")

total = labeled.height
print(f"{'class':>8s}  {'count':>8s}  {'pct':>6s}  {'hit_bar µ':>10s}  {'hit_bar med':>11s}  {'ret µ':>9s}")
for row in dist.iter_rows(named=True):
    name = {1: "LONG_WIN", 0: "TIMEOUT", -1: "STOP"}[row["label"]]
    pct = 100 * row["count"] / total
    hbm = f"{row['hit_bar_mean']:.1f}" if row["hit_bar_mean"] is not None else "-"
    hmm = f"{row['hit_bar_median']:.0f}" if row["hit_bar_median"] is not None else "-"
    rm = f"{row['ret_mean']*100:+.2f}%" if row["ret_mean"] is not None else "-"
    print(f"{name:>8s}  {row['count']:>8d}  {pct:>5.1f}%  {hbm:>10s}  {hmm:>11s}  {rm:>9s}")

# %% [markdown]
# ## 4. Validação — simetria
# Em mercado sem drift, deveríamos ter long_win ≈ stop com horizonte simétrico.
# BTC tem drift positivo, então long_win deve ser ligeiramente > stop.

# %%
long_w = labeled.filter(pl.col("label") == 1).height
stop = labeled.filter(pl.col("label") == -1).height
timeout = labeled.filter(pl.col("label") == 0).height
print(f"long_win/stop ratio: {long_w/stop:.3f}")
print(f"timeout %:           {100*timeout/total:.1f}%  (idealmente 30-60%)")

# %% [markdown]
# ## 5. Hit time — quanto rápido o mercado bate a barreira

# %%
hit = labeled.filter(pl.col("label") != 0)["hit_bar"].drop_nulls()
print(f"Hit time (em bars 15m):")
print(f"  count:    {len(hit)}")
print(f"  mean:     {hit.mean():.1f}  ({hit.mean()*15/60:.1f}h)")
print(f"  median:   {hit.median():.0f}  ({hit.median()*15/60:.1f}h)")
print(f"  p25/p75:  {hit.quantile(0.25):.0f} / {hit.quantile(0.75):.0f}")

# %% [markdown]
# ## 6. Por regime — distribuição em bull vs bear
# Define regime simples: MA 90d crescente = bull, decrescente = bear.

# %%
reg = labeled.with_columns(
    (pl.col("ma_90d").diff(96 * 7) > 0).alias("is_bull")
).drop_nulls("is_bull")

print("Distribuição em BULL:")
b = reg.filter(pl.col("is_bull"))
print(b.group_by("label").agg(pl.len()).sort("label"))
print("\nDistribuição em BEAR:")
br = reg.filter(~pl.col("is_bull"))
print(br.group_by("label").agg(pl.len()).sort("label"))

# %% [markdown]
# ## 7. Salva matrix final pra modelo
# Junta features + label num único parquet pra próximo notebook.

# %%
keep_features = [c for c in labeled.columns if c not in feat.LAG_SAFE_EXCLUDE and c not in {"label", "hit_bar", "barrier_ret", "upper_px", "lower_px"}]
# Core features que SEMPRE existem (price/técnico). Sentiment/macro/funding podem ser NaN
# em períodos antigos — LightGBM lida com NaN, dropar perderia 4 anos de OHLCV.
core_features = [c for c in keep_features if c.startswith(("ret_", "rv_", "rsi_", "ma_", "dist_", "vol_z", "atr_", "bb_", "logret"))]
final = labeled.select([
    "open_time", "close", "label", "hit_bar", "barrier_ret",
    *keep_features,
]).drop_nulls(subset=core_features + ["label"])

out_path = ROOT / "data" / "training_matrix.parquet"
final.write_parquet(out_path)
print(f"\nSalvo em {out_path}")
print(f"  shape: {final.shape}")
print(f"  features: {len(keep_features)}")
print(f"  range: bar {final['open_time'].min()} → {final['open_time'].max()}")
