# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # 02 — Baseline burro
#
# Antes de qualquer ML, estabelece o piso. Se o modelo sofisticado não bate isso,
# tem bug ou não tem edge.
#
# Estratégias testadas:
#   1. Buy-and-hold (BTC simples)
#   2. RSI<30 compra / RSI>70 vende  (mean-reversion clássico)
#   3. Random direcional (controle estatístico)
#   4. Momentum 4h (top 25% de retorno = long, bottom 25% = short)
#
# Custos: taker 0.05% + slippage 0.03% = 0.08% por trade entrada+saída
# Métricas: Sharpe anualizado, max drawdown, profit factor, win rate

# %%
from __future__ import annotations
from pathlib import Path
import polars as pl
import numpy as np

DATA = Path("../data") if Path("../data").exists() else Path("data")
BARS_PER_YEAR = 96 * 365  # velas 15m
COST_ROUND = 0.0008  # 0.08% round-trip

ohlcv = pl.read_parquet(DATA / "ohlcv_15m.parquet").sort("open_time")
o = ohlcv.with_columns(
    pl.from_epoch(pl.col("open_time") // 1000, time_unit="s").alias("ts"),
    pl.col("close").pct_change().alias("ret"),
).drop_nulls("ret")
print(f"Universo: {o.height} velas, {o['ts'].min()} → {o['ts'].max()}")


# %%
def metrics(returns: np.ndarray, signal: np.ndarray, label: str) -> dict:
    """Aplica sinal (shift 1 pra não usar info contemporânea) e mede.

    `signal` ∈ {-1, 0, +1}. Custo cobrado a cada mudança de posição.
    """
    pos = np.roll(signal, 1)
    pos[0] = 0
    changes = np.abs(np.diff(pos, prepend=0))
    costs = changes * (COST_ROUND / 2)  # cada lado paga metade
    strat_ret = pos * returns - costs

    eq = np.cumprod(1 + strat_ret)
    total = eq[-1] - 1
    n = len(strat_ret)
    years = n / BARS_PER_YEAR
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else 0

    mu = strat_ret.mean()
    sd = strat_ret.std()
    sharpe = (mu / sd) * np.sqrt(BARS_PER_YEAR) if sd > 0 else 0

    # Max drawdown
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / running_max
    max_dd = dd.min()

    # Profit factor sobre trades não-zero
    pnl = strat_ret[strat_ret != 0]
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    pf = gains / losses if losses > 0 else np.inf

    win_rate = (pnl > 0).mean() if len(pnl) else 0
    trades = int(changes.sum())

    return {
        "label": label,
        "total": total,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "profit_factor": pf,
        "win_rate": win_rate,
        "trades": trades,
        "exposure": (pos != 0).mean(),
    }


def show(rows):
    print(
        f"{'estratégia':<28s}  {'total':>9s}  {'cagr':>8s}  {'sharpe':>7s}  "
        f"{'maxDD':>8s}  {'PF':>6s}  {'win%':>6s}  {'trades':>7s}  {'expo':>6s}"
    )
    for r in rows:
        print(
            f"{r['label']:<28s}  {r['total']*100:>+8.1f}%  {r['cagr']*100:>+7.1f}%  "
            f"{r['sharpe']:>7.2f}  {r['max_dd']*100:>+7.1f}%  {r['profit_factor']:>6.2f}  "
            f"{r['win_rate']*100:>5.1f}%  {r['trades']:>7d}  {r['exposure']*100:>5.1f}%"
        )


# %% [markdown]
# ## 1. Buy & hold

# %%
returns = o["ret"].to_numpy()
n = len(returns)
results = []

bh = np.ones(n, dtype=int)  # sempre long
results.append(metrics(returns, bh, "Buy-and-hold"))


# %% [markdown]
# ## 2. RSI 14 (em velas 15m) — <30 compra, >70 vende

# %%
def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    # média EMA simplificada
    avg_g = np.zeros_like(close)
    avg_l = np.zeros_like(close)
    avg_g[:period] = np.mean(gain[:period])
    avg_l[:period] = np.mean(loss[:period])
    for i in range(period, len(close)):
        avg_g[i] = (avg_g[i - 1] * (period - 1) + gain[i]) / period
        avg_l[i] = (avg_l[i - 1] * (period - 1) + loss[i]) / period
    rs = avg_g / np.where(avg_l == 0, 1e-12, avg_l)
    return 100 - 100 / (1 + rs)


close = o["close"].to_numpy()
rsi_arr = rsi(close, 14)

# Sinal binário: long quando RSI<30, short quando RSI>70, flat caso contrário
sig = np.where(rsi_arr < 30, 1, np.where(rsi_arr > 70, -1, 0))
results.append(metrics(returns, sig, "RSI<30 L / RSI>70 S"))

# Variação só long (sem shortar topo)
sig_long = np.where(rsi_arr < 30, 1, 0)
# Hold até RSI subir acima de 50
pos = np.zeros(n, dtype=int)
holding = 0
for i in range(n):
    if rsi_arr[i] < 30:
        holding = 1
    elif rsi_arr[i] > 50:
        holding = 0
    pos[i] = holding
results.append(metrics(returns, pos, "RSI<30 buy, exit RSI>50"))


# %% [markdown]
# ## 3. Random direcional (controle)

# %%
rng = np.random.default_rng(seed=42)
random_sig = rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3])
results.append(metrics(returns, random_sig, "Random direcional"))


# %% [markdown]
# ## 4. Momentum 4h (top quartil long, bottom quartil short)

# %%
ret_4h = o["close"].pct_change(16).to_numpy()  # 16 velas = 4h
ret_4h = np.nan_to_num(ret_4h)
# Quartis rolling 30 dias (96*30=2880 velas)
window = 2880
q_hi = np.full(n, np.nan)
q_lo = np.full(n, np.nan)
for i in range(window, n):
    w = ret_4h[i - window : i]
    q_hi[i] = np.quantile(w, 0.75)
    q_lo[i] = np.quantile(w, 0.25)
sig_mom = np.where(ret_4h > q_hi, 1, np.where(ret_4h < q_lo, -1, 0))
sig_mom[np.isnan(q_hi)] = 0
results.append(metrics(returns, sig_mom, "Momentum 4h (quartil 30d)"))


# %% [markdown]
# ## Comparativo final

# %%
show(results)

# %% [markdown]
# ## Leitura honesta
#
# - Se o LightGBM da Fase 5 não bater pelo menos o Sharpe do Buy-and-Hold líquido de custos,
#   o modelo não tem edge — só ruído.
# - RSI sozinho costuma ser breakeven ou negativo em BTC depois de custos. Esperado.
# - Random + flat caro: PF próximo de 1 e Sharpe ~0. Confirma sanidade do framework.
# - Momentum 4h pode parecer bom em CAGR mas exige análise por regime — bull/chop/bear separado.
