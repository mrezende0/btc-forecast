"""exp_backtest_1k — Backtest realista DUAL-HORIZON com $1000.

Regras:
  - Capital inicial $1000, posição única, 100% notional alocado em cada trade
  - Sinal: ambas probas (mid h=12 e long h=18) > 0.35
  - Triple-barrier ±3×ATR, timeout=12 bars (48h, horizonte do modelo mid)
  - Custo round-trip 0.0008 (8 bps)
  - Walk-forward com RETREINO a cada 90 dias usando TODO histórico até t (purge HORIZON_BARS=12)
  - Causal: na barra t, modelo só vê features computadas em t-1 (lag=1) e treina até t-12

Saídas:
  - data/backtest_equity.parquet (ts, capital, drawdown)
  - relatório no stdout (tabela de checkpoints, métricas, comparação buy&hold, Sharpe)
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

# ----------------------------------------------------------------- parâmetros
TIMEFRAME_MIN = 240          # 4h
HORIZON_MID = 12             # 48h
HORIZON_LONG = 18            # 72h
ATR_MULT = 3.0
SIGNAL_THRESHOLD = 0.35
COST = 0.0008                # round-trip
BARS_PER_DAY = 6             # 24/4
RETRAIN_EVERY_BARS = 90 * BARS_PER_DAY   # ~90 dias = 540 bars 4h
START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)
INITIAL_CAPITAL = 1000.0

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


# ----------------------------------------------------------------- helpers
def build_matrix(horizon_bars: int) -> tuple[pd.DataFrame, list[str]]:
    df = feat.build_v2_from_parquets(timeframe_min=TIMEFRAME_MIN, lag=1).drop_nulls(subset=["atr_14"])
    labeled = lab.triple_barrier(df, upper_mult=ATR_MULT, lower_mult=ATR_MULT, horizon_bars=horizon_bars)
    labeled = labeled.with_columns((pl.col("label") == 1).cast(pl.Int8).alias("y"))
    fc = [
        c for c in labeled.columns
        if c not in feat.LAG_SAFE_EXCLUDE
        and c not in {"label", "hit_bar", "barrier_ret", "upper_px", "lower_px", "y"}
    ]
    # atr_14 já está nas features; manter OHLC pra simulação de barreiras
    keep = ["open_time", "open", "high", "low", "close", "y", *fc]
    mat = labeled.select(keep).drop_nulls(subset=fc + ["y"]).to_pandas()
    return mat, fc


def fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def fmt_pct(x: float) -> str:
    return f"{100*x:+.2f}%"


# ----------------------------------------------------------------- main
def main() -> None:
    print(">>> construindo matrizes (h=12 e h=18) ...")
    t0 = time.time()
    mat_mid, fc_mid = build_matrix(HORIZON_MID)
    mat_long, fc_long = build_matrix(HORIZON_LONG)
    print(f"  mid: {len(mat_mid):,} linhas, {len(fc_mid)} features")
    print(f"  long: {len(mat_long):,} linhas, {len(fc_long)} features")
    print(f"  ({time.time() - t0:.1f}s)")

    # alinha os dois pela open_time -> precisamos predizer em cada bar a partir de 2023-01-01
    # usaremos mat_mid como índice mestre; pra cada t, busca a row correspondente em mat_long
    mat_mid["dt"] = pd.to_datetime(mat_mid["open_time"], unit="ms", utc=True)
    mat_long["dt"] = pd.to_datetime(mat_long["open_time"], unit="ms", utc=True)
    mat_long_idx = mat_long.set_index("open_time")

    start_ms = int(START_DATE.timestamp() * 1000)
    start_pos = mat_mid["open_time"].searchsorted(start_ms)
    print(f">>> simulação inicia em pos={start_pos} ({mat_mid.iloc[start_pos]['dt']})")
    print(f"    fim em pos={len(mat_mid) - 1} ({mat_mid.iloc[-1]['dt']})")

    n_bars = len(mat_mid)
    capital = INITIAL_CAPITAL
    peak = capital
    in_position = False
    entry_idx = -1
    entry_px = np.nan
    target_px = np.nan
    stop_px = np.nan
    expiry_idx = -1

    equity_ts: list[int] = []
    equity_val: list[float] = []
    equity_dd: list[float] = []

    trades: list[dict] = []

    # cache modelos
    model_mid: lgb.Booster | None = None
    model_long: lgb.Booster | None = None
    last_train_idx = -10**9

    # ndarrays pra velocidade
    fc_mid_arr = mat_mid[fc_mid].to_numpy()
    fc_long_df = mat_long.set_index("open_time")[fc_long]
    open_times = mat_mid["open_time"].to_numpy()
    closes = mat_mid["close"].to_numpy()
    highs = mat_mid["high"].to_numpy()
    lows = mat_mid["low"].to_numpy()
    atrs = mat_mid["atr_14"].to_numpy()
    y_mid_arr = mat_mid["y"].to_numpy()
    y_long_arr = mat_long.set_index("open_time")["y"]

    t_loop = time.time()
    for i in range(start_pos, n_bars):
        # ----- RETREINO walk-forward -----
        if model_mid is None or (i - last_train_idx) >= RETRAIN_EVERY_BARS:
            # treino mid: rows [0, i-HORIZON_MID) — todas têm label já realizado
            cut_mid = i - HORIZON_MID
            if cut_mid > 500:
                X_tr = fc_mid_arr[:cut_mid]
                y_tr = y_mid_arr[:cut_mid]
                model_mid = lgb.train(
                    LGB_PARAMS, lgb.Dataset(X_tr, y_tr), num_boost_round=N_ROUNDS
                )
                # treino long: usa mat_long alinhado; pegar rows com open_time < open_times[i-HORIZON_LONG]
                cutoff_ot = open_times[i - HORIZON_LONG] if i - HORIZON_LONG >= 0 else open_times[0]
                mask_long = mat_long["open_time"] < cutoff_ot
                X_trl = mat_long.loc[mask_long, fc_long].to_numpy()
                y_trl = mat_long.loc[mask_long, "y"].to_numpy()
                if len(X_trl) > 500:
                    model_long = lgb.train(
                        LGB_PARAMS, lgb.Dataset(X_trl, y_trl), num_boost_round=N_ROUNDS
                    )
                    last_train_idx = i
                    elapsed = time.time() - t_loop
                    print(
                        f"  [retreino] bar={i} dt={mat_mid.iloc[i]['dt'].strftime('%Y-%m-%d')} "
                        f"n_tr_mid={cut_mid:,} n_tr_long={mask_long.sum():,} "
                        f"capital={fmt_money(capital)} ({elapsed:.0f}s)"
                    )

        # ----- gerencia posição existente -----
        if in_position:
            hit_target = highs[i] >= target_px
            hit_stop = lows[i] <= stop_px
            timed_out = i >= expiry_idx

            exit_now = False
            exit_px = np.nan
            reason = ""
            # conservador: se ambos no mesmo bar, assume stop primeiro (pior caso)
            if hit_stop:
                exit_px = stop_px
                reason = "stop"
                exit_now = True
            elif hit_target:
                exit_px = target_px
                reason = "target"
                exit_now = True
            elif timed_out:
                exit_px = closes[i]
                reason = "timeout"
                exit_now = True

            if exit_now:
                gross_ret = exit_px / entry_px - 1
                net_ret = gross_ret - COST
                capital *= (1 + net_ret)
                trades.append(
                    {
                        "entry_dt": mat_mid.iloc[entry_idx]["dt"],
                        "exit_dt": mat_mid.iloc[i]["dt"],
                        "entry_px": entry_px,
                        "exit_px": exit_px,
                        "gross_ret": gross_ret,
                        "net_ret": net_ret,
                        "reason": reason,
                        "bars_held": i - entry_idx,
                        "capital_after": capital,
                    }
                )
                in_position = False

        # ----- avalia sinal pra próxima posição (só se flat) -----
        if not in_position and model_mid is not None and model_long is not None:
            x_mid = fc_mid_arr[i : i + 1]
            proba_mid = float(model_mid.predict(x_mid)[0])
            ot_i = open_times[i]
            if ot_i in fc_long_df.index:
                x_long = fc_long_df.loc[[ot_i]].to_numpy()
                proba_long = float(model_long.predict(x_long)[0])
                signal = (proba_mid > SIGNAL_THRESHOLD) and (proba_long > SIGNAL_THRESHOLD)
                if signal and not np.isnan(atrs[i]):
                    # ENTRADA: a barra t fecha; assumimos entrada no close[t]
                    entry_idx = i
                    entry_px = closes[i]
                    target_px = entry_px + ATR_MULT * atrs[i]
                    stop_px = entry_px - ATR_MULT * atrs[i]
                    expiry_idx = i + HORIZON_MID
                    in_position = True

        # ----- equity mark-to-market -----
        if in_position:
            # marcar pelo close atual
            unreal = closes[i] / entry_px - 1
            mtm = capital * (1 + unreal - COST)  # já desconta custo
        else:
            mtm = capital
        peak = max(peak, mtm)
        dd = mtm / peak - 1
        equity_ts.append(int(open_times[i]))
        equity_val.append(mtm)
        equity_dd.append(dd)

    # se terminar em posição, fecha no último close
    if in_position:
        exit_px = closes[-1]
        gross_ret = exit_px / entry_px - 1
        net_ret = gross_ret - COST
        capital *= (1 + net_ret)
        trades.append(
            {
                "entry_dt": mat_mid.iloc[entry_idx]["dt"],
                "exit_dt": mat_mid.iloc[-1]["dt"],
                "entry_px": entry_px,
                "exit_px": exit_px,
                "gross_ret": gross_ret,
                "net_ret": net_ret,
                "reason": "forced_close_end",
                "bars_held": (n_bars - 1) - entry_idx,
                "capital_after": capital,
            }
        )

    # ----------------------------------------------------------- relatório
    eq_df = pd.DataFrame(
        {
            "ts": pd.to_datetime(equity_ts, unit="ms", utc=True),
            "capital": equity_val,
            "drawdown": equity_dd,
        }
    )
    out_path = ROOT / "data" / "backtest_equity.parquet"
    pl.from_pandas(eq_df).write_parquet(out_path)
    print(f"\n>>> equity salva em {out_path} ({len(eq_df):,} pontos)")

    trades_df = pd.DataFrame(trades)
    n_trades = len(trades_df)
    wins = (trades_df["net_ret"] > 0).sum() if n_trades else 0
    win_rate = wins / n_trades if n_trades else 0
    avg_pnl = trades_df["net_ret"].mean() if n_trades else 0
    avg_bars = trades_df["bars_held"].mean() if n_trades else 0

    # max DD & % tempo em DD
    max_dd = eq_df["drawdown"].min()
    pct_in_dd = (eq_df["drawdown"] < -0.005).mean()

    # Sharpe anualizado — usa retornos por bar (4h => sqrt(6*365))
    eq_df["ret"] = eq_df["capital"].pct_change().fillna(0)
    bars_per_year = 6 * 365
    if eq_df["ret"].std() > 0:
        sharpe = (eq_df["ret"].mean() / eq_df["ret"].std()) * np.sqrt(bars_per_year)
    else:
        sharpe = 0.0

    # buy & hold no mesmo período: comprado em closes[start_pos], vendido no último close
    bh_entry = closes[start_pos]
    bh_exit = closes[-1]
    bh_ret = bh_exit / bh_entry - 1
    bh_final = INITIAL_CAPITAL * (1 + bh_ret - COST)

    # checkpoints
    start_dt = eq_df.iloc[0]["ts"]
    checkpoints = [
        ("1 semana", pd.Timedelta(days=7)),
        ("1 mês", pd.Timedelta(days=30)),
        ("3 meses", pd.Timedelta(days=90)),
        ("6 meses", pd.Timedelta(days=180)),
        ("1 ano", pd.Timedelta(days=365)),
        ("2 anos", pd.Timedelta(days=2 * 365)),
        ("3 anos", pd.Timedelta(days=3 * 365)),
    ]

    print("\n" + "=" * 72)
    print(" RELATÓRIO DE BACKTEST — DUAL-HORIZON ($1000 inicial)")
    print("=" * 72)
    print(f" Período:      {start_dt.strftime('%Y-%m-%d')} → {eq_df.iloc[-1]['ts'].strftime('%Y-%m-%d')}")
    print(f" Bars 4h:      {len(eq_df):,}  ({len(eq_df)/BARS_PER_DAY:.0f} dias)")
    print()

    print(" Checkpoints de capital:")
    print(f"  {'horizonte':<12s} {'data':<12s} {'capital':>12s} {'retorno':>10s} {'BTC$':>10s} {'BH cap':>12s}")
    btc_start = closes[start_pos]
    for label, delta in checkpoints:
        target_dt = start_dt + delta
        sub = eq_df[eq_df["ts"] <= target_dt]
        if sub.empty:
            continue
        row = sub.iloc[-1]
        # buy & hold no mesmo instante
        ot_target_ms = int(row["ts"].timestamp() * 1000)
        idx_close = np.searchsorted(open_times, ot_target_ms)
        idx_close = min(idx_close, len(closes) - 1)
        btc_px = closes[idx_close]
        bh_cap = INITIAL_CAPITAL * (btc_px / btc_start) * (1 - COST)
        print(
            f"  {label:<12s} {row['ts'].strftime('%Y-%m-%d')} "
            f"{fmt_money(row['capital']):>12s} {fmt_pct(row['capital']/INITIAL_CAPITAL-1):>10s} "
            f"${btc_px:>8,.0f} {fmt_money(bh_cap):>12s}"
        )
    # FINAL
    row = eq_df.iloc[-1]
    print(
        f"  {'FINAL':<12s} {row['ts'].strftime('%Y-%m-%d')} "
        f"{fmt_money(row['capital']):>12s} {fmt_pct(row['capital']/INITIAL_CAPITAL-1):>10s} "
        f"${closes[-1]:>8,.0f} {fmt_money(bh_final):>12s}"
    )

    print()
    print(" Estatísticas de trades:")
    print(f"  Total trades:           {n_trades}")
    print(f"  Win rate:               {100*win_rate:.1f}%")
    print(f"  Avg PnL líquido/trade:  {100*avg_pnl:+.2f}%")
    print(f"  Avg bars em posição:    {avg_bars:.1f} ({avg_bars*4:.0f}h)")
    if n_trades:
        wins_avg = trades_df.loc[trades_df["net_ret"] > 0, "net_ret"].mean()
        losses_avg = trades_df.loc[trades_df["net_ret"] <= 0, "net_ret"].mean()
        print(f"  Avg win:                {100*wins_avg:+.2f}%")
        print(f"  Avg loss:               {100*losses_avg:+.2f}%")
        by_reason = trades_df["reason"].value_counts()
        print(f"  Saídas por razão:       {by_reason.to_dict()}")

    print()
    print(" Risco:")
    print(f"  Max drawdown:           {100*max_dd:.2f}%")
    print(f"  % tempo em DD (<-0.5%): {100*pct_in_dd:.1f}%")
    print(f"  Sharpe anualizado:      {sharpe:.2f}")

    print()
    print(" Comparação Buy & Hold ($1000 mesmo período):")
    print(f"  BH final:               {fmt_money(bh_final)}  ({fmt_pct(bh_final/INITIAL_CAPITAL-1)})")
    print(f"  Modelo final:           {fmt_money(capital)}  ({fmt_pct(capital/INITIAL_CAPITAL-1)})")
    edge = capital - bh_final
    print(f"  Edge vs BH:             {fmt_money(edge)} ({fmt_pct(edge/bh_final)})")

    # gráfico ASCII simples — capital + drawdown
    print()
    print(" Equity curve (ASCII, normalizado):")
    n_buckets = 60
    step = max(1, len(eq_df) // n_buckets)
    samples = eq_df.iloc[::step].reset_index(drop=True)
    mn, mx = samples["capital"].min(), samples["capital"].max()
    rng = mx - mn or 1
    height = 12
    grid = [[" "] * len(samples) for _ in range(height)]
    for x, v in enumerate(samples["capital"]):
        y = int((v - mn) / rng * (height - 1))
        y = height - 1 - y
        grid[y][x] = "*"
    for row in grid:
        print("  " + "".join(row))
    print(f"  min={fmt_money(mn)}  max={fmt_money(mx)}  bars={len(samples)}")

    print()
    print(" Recomendação:")
    sharpe_ok = sharpe > 1.0
    beats_bh = capital > bh_final
    dd_ok = max_dd > -0.30
    enough_trades = n_trades >= 30
    verdicts = []
    if sharpe_ok:
        verdicts.append(f"Sharpe {sharpe:.2f} > 1.0 OK")
    else:
        verdicts.append(f"Sharpe {sharpe:.2f} <= 1.0 BAIXO")
    if beats_bh:
        verdicts.append(f"bateu BH em {fmt_money(edge)}")
    else:
        verdicts.append(f"perdeu pra BH em {fmt_money(-edge)}")
    if dd_ok:
        verdicts.append(f"DD {100*max_dd:.1f}% gerenciável")
    else:
        verdicts.append(f"DD {100*max_dd:.1f}% alto")
    verdicts.append(f"{n_trades} trades (amostra {'suficiente' if enough_trades else 'fraca'})")
    print("  " + " | ".join(verdicts))

    score = sum([sharpe_ok, beats_bh, dd_ok, enough_trades])
    if score >= 3 and beats_bh and sharpe_ok:
        print("  >>> VEREDITO: modelo MOSTRA EDGE estatístico — vale operar com sizing reduzido.")
    elif beats_bh and sharpe > 0.5:
        print("  >>> VEREDITO: modelo MARGINAL — viável só com gestão de risco apertada / sizing baixo.")
    else:
        print("  >>> VEREDITO: modelo NÃO compensa o risco vs buy & hold no período. Não operar como está.")


if __name__ == "__main__":
    main()
