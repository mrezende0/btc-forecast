# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # 01 — EDA inicial
#
# Objetivo: entender o que temos em mãos antes de pensar em modelagem.
# Foco em OHLCV 15m + funding + macro + F&G. Notícias entram depois do backfill GDELT.
#
# Pra abrir como notebook:
#   pip install jupytext   # já está em requirements-dev
#   jupytext --to notebook notebooks/01_eda.py
# Ou abrir direto no VSCode (rende cells `# %%` automaticamente).

# %%
from __future__ import annotations
from pathlib import Path
import polars as pl
import numpy as np

DATA = Path("../data") if Path("../data").exists() else Path("data")

# %% [markdown]
# ## 1. Inventário

# %%
ohlcv = pl.read_parquet(DATA / "ohlcv_15m.parquet")
funding = pl.read_parquet(DATA / "funding.parquet")
macro = pl.read_parquet(DATA / "macro_daily.parquet")
fg = pl.read_parquet(DATA / "fg_daily.parquet")

for name, df in [("ohlcv_15m", ohlcv), ("funding", funding), ("macro", macro), ("fg", fg)]:
    print(f"{name:14s}  rows={df.height:>7}  cols={df.columns}")

# %% [markdown]
# ## 2. Range temporal + gaps

# %%
o = ohlcv.with_columns(
    pl.from_epoch(pl.col("open_time") // 1000, time_unit="s").alias("ts")
).sort("ts")
print(f"OHLCV: {o['ts'].min()} → {o['ts'].max()}  ({o.height} velas)")

diffs = o.with_columns(
    (pl.col("open_time") - pl.col("open_time").shift(1)).alias("dt_ms")
)
expected = 15 * 60 * 1000
gaps = diffs.filter(pl.col("dt_ms") > expected)
print(f"Gaps > 15min: {gaps.height}")
if gaps.height:
    print("Top 5 maiores gaps:")
    print(gaps.sort("dt_ms", descending=True).select(["ts", "dt_ms"]).head(5))

# %% [markdown]
# ## 3. Retornos 15m — distribuição e estatísticas

# %%
o = o.with_columns(
    pl.col("close").pct_change().alias("ret_15m"),
    (pl.col("close").log() - pl.col("close").shift(1).log()).alias("logret_15m"),
)
r = o["logret_15m"].drop_nulls()
print(f"N retornos: {len(r)}")
print(f"  mean:   {r.mean():.6f}")
print(f"  std:    {r.std():.6f}")
print(f"  skew:   {r.skew():.4f}")
print(f"  kurt:   {r.kurtosis():.4f}  (>0 = fat tails)")
print(f"  q01/q99: {r.quantile(0.01):.4f} / {r.quantile(0.99):.4f}")
print(f"  min/max: {r.min():.4f} / {r.max():.4f}")

# Vol anualizada
ann = r.std() * np.sqrt(96 * 365)  # 96 velas/dia
print(f"  vol anualizada: {ann*100:.1f}%")

# %% [markdown]
# ## 4. Autocorrelação dos retornos
# Quanto MAIS perto de zero, MAIS difícil de prever direção. Negativo em curto prazo
# = mean reversion (microestrutura). Positivo = momentum.

# %%
import polars as pl
r_arr = r.to_numpy()
for lag in [1, 4, 16, 96, 96*7]:
    if len(r_arr) > lag:
        ac = np.corrcoef(r_arr[:-lag], r_arr[lag:])[0, 1]
        label = {1:"15min", 4:"1h", 16:"4h", 96:"1d", 96*7:"1sem"}.get(lag, f"{lag}lag")
        print(f"  ρ(t, t-{label:5s}) = {ac:+.4f}")

# %% [markdown]
# ## 5. Volatilidade rolling (regime)
# Vol realizada 1d, mostra quando o mercado fica calmo vs explosivo.

# %%
rv = (
    o.with_columns(
        pl.col("logret_15m").rolling_std(window_size=96).alias("rv_1d")
    )
    .with_columns((pl.col("rv_1d") * np.sqrt(96 * 365) * 100).alias("rv_ann_pct"))
    .drop_nulls("rv_ann_pct")
)
rv_arr = rv["rv_ann_pct"].to_numpy()
print(f"Vol anualizada rolling 1d:")
print(f"  mediana: {np.median(rv_arr):.1f}%")
print(f"  p25/p75: {np.percentile(rv_arr,25):.1f}% / {np.percentile(rv_arr,75):.1f}%")
print(f"  p05/p95: {np.percentile(rv_arr,5):.1f}% / {np.percentile(rv_arr,95):.1f}%")

# %% [markdown]
# ## 6. Sazonalidade — hora-do-dia e dia-da-semana

# %%
season = o.with_columns(
    pl.col("ts").dt.hour().alias("hour"),
    pl.col("ts").dt.weekday().alias("dow"),
).drop_nulls("logret_15m")

hourly = (
    season.group_by("hour")
    .agg(
        pl.col("logret_15m").mean().alias("mean_ret"),
        pl.col("logret_15m").std().alias("std_ret"),
        pl.len().alias("n"),
    )
    .sort("hour")
)
print("Por hora UTC (média de retorno 15m × 10000):")
for row in hourly.iter_rows(named=True):
    bar = "▲" if row["mean_ret"] > 0 else "▼"
    print(f"  {row['hour']:02d}:00  {row['mean_ret']*10000:+6.2f}  vol={row['std_ret']*100:.3f}%  n={row['n']}  {bar}")

print()
dow_map = {1:"Seg", 2:"Ter", 3:"Qua", 4:"Qui", 5:"Sex", 6:"Sáb", 7:"Dom"}
weekly = (
    season.group_by("dow")
    .agg(
        pl.col("logret_15m").mean().alias("mean_ret"),
        pl.col("logret_15m").std().alias("std_ret"),
        pl.len().alias("n"),
    )
    .sort("dow")
)
print("Por dia da semana (mean × 10000):")
for row in weekly.iter_rows(named=True):
    print(f"  {dow_map.get(row['dow'],row['dow'])}  {row['mean_ret']*10000:+6.2f}  vol={row['std_ret']*100:.3f}%")

# %% [markdown]
# ## 7. Funding rate
# Em mercado normal está perto de +0.01% (8h). Spikes positivos = mercado long demais
# (reversão pra baixo provável). Negativos extremos = capitulação short.

# %%
f = funding.with_columns(
    pl.from_epoch(pl.col("funding_time") // 1000, time_unit="s").alias("ts")
).sort("ts")
print(f"Funding: {f['ts'].min()} → {f['ts'].max()}  ({f.height} pontos)")
print(f"  mean:   {f['funding_rate'].mean()*100:.4f}%")
print(f"  std:    {f['funding_rate'].std()*100:.4f}%")
print(f"  q05/q95: {f['funding_rate'].quantile(0.05)*100:.4f}% / {f['funding_rate'].quantile(0.95)*100:.4f}%")
print(f"  min/max: {f['funding_rate'].min()*100:.4f}% / {f['funding_rate'].max()*100:.4f}%")

# %% [markdown]
# ## 8. Macro — DXY, VIX, SPX
# Pra ver se faz sentido como contexto. Não esperamos correlação fortíssima intraday,
# mas em janelas diárias macro pesa.

# %%
m = macro.drop_nulls()
print(f"Macro: {m['date'].min()} → {m['date'].max()}  ({m.height} dias)")

# Daily BTC close pra correlacionar
btc_daily = (
    o.group_by(pl.col("ts").dt.date().alias("date"))
    .agg(pl.col("close").last().alias("btc_close"))
    .sort("date")
)
joined = btc_daily.join(m, on="date", how="inner").drop_nulls()
# Retornos diários
joined = joined.with_columns(
    pl.col("btc_close").pct_change().alias("btc_ret"),
    pl.col("dxy").pct_change().alias("dxy_ret"),
    pl.col("vix").pct_change().alias("vix_ret"),
    pl.col("spx").pct_change().alias("spx_ret"),
).drop_nulls()

print(f"Janela conjunta: {joined.height} dias")
print("Correlação retornos diários:")
for col in ["dxy_ret", "vix_ret", "spx_ret"]:
    c = np.corrcoef(joined["btc_ret"].to_numpy(), joined[col].to_numpy())[0, 1]
    print(f"  BTC vs {col[:3].upper()}:  {c:+.3f}")

# %% [markdown]
# ## 9. Fear & Greed
# Indicador composto. Extremos (>75 ou <25) são historicamente bons sinais contrarian.

# %%
print(f"F&G: {fg['date'].min()} → {fg['date'].max()}  ({fg.height} dias)")
print(f"  mean: {fg['fg_value'].mean():.1f}  std: {fg['fg_value'].std():.1f}")
print(f"  dist: min={fg['fg_value'].min()}  p25={fg['fg_value'].quantile(0.25):.0f}  median={fg['fg_value'].median():.0f}  p75={fg['fg_value'].quantile(0.75):.0f}  max={fg['fg_value'].max()}")
print(f"  classes: {fg.group_by('fg_class').len().sort('len', descending=True).to_dicts()}")

# Cruzamento F&G com retorno 7d forward
fg_btc = fg.join(btc_daily, on="date", how="inner").sort("date")
fg_btc = fg_btc.with_columns(
    (pl.col("btc_close").shift(-7) / pl.col("btc_close") - 1).alias("fwd_ret_7d")
).drop_nulls("fwd_ret_7d")

print("\nRetorno forward 7d médio por bin de F&G:")
fg_btc = fg_btc.with_columns(
    pl.when(pl.col("fg_value") < 25).then(pl.lit("0-25 (ext fear)"))
    .when(pl.col("fg_value") < 50).then(pl.lit("25-50 (fear)"))
    .when(pl.col("fg_value") < 75).then(pl.lit("50-75 (greed)"))
    .otherwise(pl.lit("75+ (ext greed)")).alias("bin")
)
binned = fg_btc.group_by("bin").agg(
    pl.col("fwd_ret_7d").mean().alias("mean"),
    pl.col("fwd_ret_7d").median().alias("median"),
    pl.col("fwd_ret_7d").std().alias("std"),
    pl.len().alias("n"),
)
for row in binned.iter_rows(named=True):
    print(f"  {row['bin']:18s}  mean={row['mean']*100:+5.2f}%  median={row['median']*100:+5.2f}%  n={row['n']}")
