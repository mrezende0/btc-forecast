# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # 03 — Feature engineering
#
# Construção e sanity-check das features. Usa `pipeline.features.build_from_parquets`.
#
# Princípios:
#   - Toda feature defasada (lag=1 vela) automaticamente via `apply_lag`
#   - Macro/F&G com available_at conservador (D+1)
#   - As-of join backward em todas as fontes daily/funding

# %%
from __future__ import annotations
from pathlib import Path
import polars as pl
import numpy as np
import sys

# Adiciona root do projeto ao path independente de CWD
ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)  # garante DATA = data/ relativo ao root

from pipeline import features as feat

# %% [markdown]
# ## 1. Build

# %%
df = feat.build_from_parquets(lag=1)
print(f"shape: {df.shape}")
print(f"colunas: {len(df.columns)}")
print(df.columns)

# %% [markdown]
# ## 2. Inventário de features (não-OHLCV)

# %%
feature_cols = [c for c in df.columns if c not in feat.LAG_SAFE_EXCLUDE]
print(f"Features: {len(feature_cols)}")
for c in feature_cols:
    null_pct = df[c].null_count() / df.height * 100
    print(f"  {c:25s}  nulls={null_pct:5.1f}%  dtype={df[c].dtype}")

# %% [markdown]
# ## 3. Sanity: nada deve ter null absurdo, exceto warm-up das janelas longas

# %%
# Linhas com TUDO presente:
clean = df.drop_nulls(subset=feature_cols)
print(f"linhas totalmente válidas: {clean.height} ({100*clean.height/df.height:.1f}%)")
print(f"perda esperada: warm-up de MA 90d (~8.6k velas), funding em 2021 antes de cobertura, etc.")

# %% [markdown]
# ## 4. Distribuição de cada feature

# %%
import builtins
desc = clean.select(feature_cols).describe()
print(desc)

# %% [markdown]
# ## 5. Correlação com retorno futuro (sanity de poder preditivo bruto)
#
# Calcula correlação de Spearman entre cada feature e retorno forward 4h / 1d.
# Esperado: valores baixos individualmente (~|0.05|), mas alguns notáveis.

# %%
# Adiciona retornos forward sem alterar features (apenas pra teste, não pra training)
work = clean.with_columns(
    (pl.col("close").shift(-feat.BARS_PER_HOUR * 4) / pl.col("close") - 1).alias("fwd_4h"),
    (pl.col("close").shift(-feat.BARS_PER_DAY) / pl.col("close") - 1).alias("fwd_1d"),
).drop_nulls(subset=["fwd_4h", "fwd_1d"])

print(f"Amostra correlação: {work.height} velas\n")

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    # Spearman via ranks
    from scipy.stats import spearmanr
    return spearmanr(a, b, nan_policy="omit").statistic

try:
    from scipy.stats import spearmanr
    has_scipy = True
except ImportError:
    has_scipy = False
    print("⚠️  scipy não disponível, usando Pearson")

print(f"{'feature':<28s}  {'corr_fwd_4h':>12s}  {'corr_fwd_1d':>12s}")
fwd_4h = work["fwd_4h"].to_numpy()
fwd_1d = work["fwd_1d"].to_numpy()
for c in feature_cols:
    x = work[c].to_numpy()
    if x.dtype == object or np.all(np.isnan(x)):
        continue
    try:
        if has_scipy:
            c4 = spearmanr(x, fwd_4h, nan_policy="omit").statistic
            c1d = spearmanr(x, fwd_1d, nan_policy="omit").statistic
        else:
            c4 = np.corrcoef(x, fwd_4h)[0, 1]
            c1d = np.corrcoef(x, fwd_1d)[0, 1]
        # Highlight quando |corr| > 0.02
        flag = " ★" if max(abs(c4 or 0), abs(c1d or 0)) > 0.02 else ""
        print(f"  {c:<28s}  {c4:>+12.4f}  {c1d:>+12.4f}{flag}")
    except Exception as e:
        print(f"  {c:<28s}  ERRO: {e}")

# %% [markdown]
# ## 6. Conferência crítica: leakage check
#
# Pra cada feature, a maior correlação |corr| deve estar com retorno FUTURO, não passado.
# Se uma feature correlaciona forte com retorno SIMULTÂNEO da vela t, suspeite de leak.

# %%
work2 = clean.with_columns(
    (pl.col("close") / pl.col("close").shift(1) - 1).alias("ret_now"),  # retorno da vela t (que o modelo "veria" se vazasse)
).drop_nulls("ret_now")

ret_now = work2["ret_now"].to_numpy()
print("Features com |corr| > 0.05 com ret_now (POTENCIAL LEAK):")
leaks = []
for c in feature_cols:
    x = work2[c].to_numpy()
    if x.dtype == object or np.all(np.isnan(x)):
        continue
    try:
        if has_scipy:
            cn = spearmanr(x, ret_now, nan_policy="omit").statistic
        else:
            cn = np.corrcoef(x, ret_now)[0, 1]
        if abs(cn) > 0.05:
            leaks.append((c, cn))
    except Exception:
        pass

if leaks:
    for c, cn in sorted(leaks, key=lambda x: -abs(x[1])):
        print(f"  ⚠️  {c:<28s}  ρ(ret_now) = {cn:+.4f}")
else:
    print("  ✓ nenhum leak suspeito.")
