# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # Comparação: Baseline vs RISK-1PCT + sem-BEAR
#
# Backtest realista lado-a-lado com posição única + compounding sequencial.

# %%
from __future__ import annotations
from pathlib import Path
import sys, os
ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timezone, timedelta
from pipeline import features as feat, labels as lab, model as mdl

# %% Config
START_CAPITAL = 1000.0
COST = 0.0015  # Binance taker 0.10% × 2 + slippage real
H_MID = 12
H_LONG = 18
ATR_MULT = 3.0
THR = 0.35
RISK_PCT = 0.01
MAX_PCT = 0.50
BARS_PER_MONTH = 180
BEAR_THR = -0.05

PARAMS = dict(objective="binary", metric="binary_logloss", verbose=-1, n_jobs=-1,
              learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, lambda_l2=0.5)

# %% Build matrizes
def setup(h):
    df = feat.build_v2_from_parquets(timeframe_min=240, lag=1).drop_nulls(subset=["atr_14"])
    lab_df = lab.triple_barrier(df, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=h)
    lab_df = lab_df.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y"))
    fcols = [c for c in lab_df.columns if c not in feat.LAG_SAFE_EXCLUDE and c not in {"label","hit_bar","barrier_ret","upper_px","lower_px","y"}]
    base_cols = ["open_time","close","high","low","y","barrier_ret"]
    extra = [c for c in fcols if c not in base_cols]
    m = lab_df.select(base_cols + extra).drop_nulls(subset=extra+["y"]).to_pandas()
    m["dt"] = m["open_time"].apply(lambda ms: datetime.fromtimestamp(ms/1000, tz=timezone.utc))
    m["quarter"] = m["dt"].dt.to_period("Q")
    return m, fcols

m_mid, fc_mid = setup(H_MID)
m_long, fc_long = setup(H_LONG)
quarters = [q for q in sorted(m_mid["quarter"].unique()) if q.start_time.year >= 2023]
print(f"mid: {len(m_mid)} rows, long: {len(m_long)} rows, {len(quarters)} quarters")

# %% Pre-compute probas walk-forward (retrain quarterly, proper purge)
all_proba_mid = np.zeros(len(m_mid))
all_proba_long = np.zeros(len(m_mid))
covered = np.zeros(len(m_mid), dtype=bool)

ot_long = {ot: i for i, ot in enumerate(m_long["open_time"].values)}

for q in quarters:
    test_idx = m_mid.index[m_mid["quarter"] == q].tolist()
    if not test_idx: continue
    train_end_mid = test_idx[0] - H_MID
    use_start = test_idx[0] + H_MID
    if train_end_mid < 500 or use_start >= test_idx[-1]: continue
    use_idx = [i for i in test_idx if i >= use_start]

    # mid model
    X_tr = m_mid.iloc[:train_end_mid][fc_mid].values
    y_tr = m_mid.iloc[:train_end_mid]["y"].values
    mm = lgb.train(PARAMS, lgb.Dataset(X_tr, y_tr), num_boost_round=500)
    X_te = m_mid.iloc[use_idx][fc_mid].values
    all_proba_mid[use_idx] = mm.predict(X_te)

    # long model
    test_idx_long = m_long.index[m_long["quarter"] == q].tolist()
    if test_idx_long:
        train_end_long = test_idx_long[0] - H_LONG
        if train_end_long >= 500:
            X_trL = m_long.iloc[:train_end_long][fc_long].values
            y_trL = m_long.iloc[:train_end_long]["y"].values
            ml = lgb.train(PARAMS, lgb.Dataset(X_trL, y_trL), num_boost_round=500)
            for i in use_idx:
                ot_val = m_mid["open_time"].iloc[i]
                j = ot_long.get(ot_val)
                if j is not None:
                    xrow = m_long.iloc[j:j+1][fc_long].values
                    all_proba_long[i] = ml.predict(xrow)[0]
    covered[use_idx] = True
    print(f"  {q} ok")

# %% Backtest engine (1 position at a time, compounding, ATR-based exits)
def backtest(sizing: str, use_bear_filter: bool, label: str):
    capital = START_CAPITAL
    cash = START_CAPITAL
    position = None  # dict or None
    trades = []
    equity = []

    df = m_mid

    for i in range(len(df)):
        if not covered[i]:
            equity.append(capital)
            continue
        row = df.iloc[i]
        ts = row["open_time"]
        close = row["close"]
        high = row["high"]
        low = row["low"]

        # ---- Manage open position
        if position is not None:
            # check stop/target on this bar
            hit_stop = low <= position["stop"]
            hit_target = high >= position["target"]
            timeout = ts >= position["timeout_at"]

            exit_price = None
            outcome = None
            if hit_stop and hit_target:
                # conservative: assume stop first
                exit_price = position["stop"]
                outcome = "stop"
            elif hit_stop:
                exit_price = position["stop"]
                outcome = "stop"
            elif hit_target:
                exit_price = position["target"]
                outcome = "target"
            elif timeout:
                exit_price = close
                outcome = "timeout"

            if exit_price is not None:
                # PnL
                pnl_pct_pos = (exit_price / position["entry"] - 1) - COST
                # capital change scaled by position fraction
                pos_value = position["size_usd"]
                pnl_usd = pos_value * pnl_pct_pos
                # update capital: position closed
                cash += pos_value + pnl_usd  # devolve principal + P&L
                capital = cash
                trades.append({
                    "entry_ts": position["entry_ts"], "exit_ts": ts,
                    "entry": position["entry"], "exit": exit_price,
                    "size_usd": pos_value, "pnl_usd": pnl_usd,
                    "pnl_pct": pnl_pct_pos, "outcome": outcome,
                    "capital_after": capital,
                })
                position = None

        # ---- Open new position
        if position is None and all_proba_mid[i] > THR and all_proba_long[i] > THR:
            # Bear filter
            if use_bear_filter and i >= BARS_PER_MONTH:
                ret_30d = close / df["close"].iloc[i - BARS_PER_MONTH] - 1
                if ret_30d < BEAR_THR:
                    equity.append(capital)
                    continue
            atr = row["atr_14"]
            entry = close
            stop = entry - ATR_MULT * atr
            target = entry + ATR_MULT * atr
            # Sizing
            if sizing == "full":
                size_usd = capital
            elif sizing == "risk1":
                risk_dollars = capital * RISK_PCT
                distance = entry - stop
                size_usd = (risk_dollars / distance) * entry
                size_usd = min(size_usd, capital * MAX_PCT)
            else:
                raise ValueError(sizing)
            cash -= size_usd
            position = {
                "entry": entry, "stop": stop, "target": target,
                "size_usd": size_usd, "entry_ts": ts,
                "timeout_at": ts + H_MID * 4 * 3600 * 1000,
            }

        # equity = cash + posição marked-to-market
        if position is not None:
            mtm = position["size_usd"] * (close / position["entry"])
            equity.append(cash + mtm)
        else:
            equity.append(capital)

    # Close any open position at end
    if position is not None:
        last_close = df["close"].iloc[-1]
        pnl_pct_pos = (last_close / position["entry"] - 1) - COST
        pos_value = position["size_usd"]
        pnl_usd = pos_value * pnl_pct_pos
        cash += pos_value + pnl_usd
        capital = cash
        trades.append({
            "entry_ts": position["entry_ts"], "exit_ts": df["open_time"].iloc[-1],
            "entry": position["entry"], "exit": last_close,
            "size_usd": pos_value, "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct_pos, "outcome": "forced",
            "capital_after": capital,
        })

    eq_arr = np.array(equity)
    return capital, eq_arr, pd.DataFrame(trades), label

# %% Run 4 combos
combos = [
    ("FULL, sem filtro",    "full",  False),
    ("FULL + sem-BEAR",     "full",  True),
    ("RISK-1PCT, sem filtro", "risk1", False),
    ("RISK-1PCT + sem-BEAR ★", "risk1", True),
]

results = []
for label, sz, bear in combos:
    cap, eq, tr, lbl = backtest(sz, bear, label)
    n = len(tr)
    win = (tr["pnl_pct"] > 0).mean() if n else 0
    avg_pnl_pct = tr["pnl_pct"].mean() if n else 0
    peak = pd.Series(eq).cummax()
    dd = (eq / peak - 1).min()
    pct_in_dd = ((eq / peak - 1) < -0.005).mean()
    ret = cap / START_CAPITAL - 1
    # Sharpe sobre retornos por bar (4h)
    eq_s = pd.Series(eq)
    bar_ret = eq_s.pct_change().fillna(0)
    sharpe = (bar_ret.mean() / bar_ret.std()) * np.sqrt(6 * 365) if bar_ret.std() > 0 else 0
    results.append(dict(label=label, capital=cap, ret=ret, n=n, win=win, avg=avg_pnl_pct,
                       sharpe=sharpe, dd=dd, pct_dd=pct_in_dd))

# %% Tabela comparativa
print(f"\n{'='*85}")
print(f"{'configuração':<32s}  {'capital':>9s}  {'ret':>7s}  {'n':>4s}  {'win%':>5s}  {'Sharpe':>7s}  {'MaxDD':>7s}  {'%DD':>5s}")
print("-"*85)
for r in results:
    print(f"  {r['label']:<30s}  ${r['capital']:>7,.0f}  {100*r['ret']:>+6.1f}%  {r['n']:>4d}  {100*r['win']:>4.1f}%  {r['sharpe']:>+6.2f}  {100*r['dd']:>+6.1f}%  {100*r['pct_dd']:>4.0f}%")

# B&H comparável: começa quando começa o backtest (1ª vela coberta)
first_cov_idx = int(np.argmax(covered))
bh_entry = m_mid["close"].iloc[first_cov_idx]
bh_final = m_mid["close"].iloc[-1]
bh_cap = START_CAPITAL * (bh_final / bh_entry)
bh_eq_arr = (m_mid["close"].iloc[first_cov_idx:] / bh_entry * START_CAPITAL).values
bh_peak = pd.Series(bh_eq_arr).cummax()
bh_dd = (bh_eq_arr / bh_peak - 1).min()
bh_ret = pd.Series(bh_eq_arr).pct_change().fillna(0)
bh_sharpe = (bh_ret.mean() / bh_ret.std()) * np.sqrt(6 * 365) if bh_ret.std() > 0 else 0
print(f"  {'Buy-and-Hold (mesmo período)':<30s}  ${bh_cap:>7,.0f}  {100*(bh_final/bh_entry - 1):>+6.1f}%   1  100.0%  {bh_sharpe:>+6.2f}  {100*bh_dd:>+6.1f}%   n/a")
print("="*85)
