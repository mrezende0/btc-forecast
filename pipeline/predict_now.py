"""Entry point chamado pelo workflow predict_4h.

Fluxo:
  1. Build matriz v2 com dados atuais
  2. Treina modelo no histórico (purge horizon)
  3. Prediz na vela mais recente (cuja barreira ainda não venceu)
  4. Loga sinal em data/signals.parquet
  5. Se proba > threshold, envia alerta Telegram

Idempotente: rerun no mesmo bar usa o mesmo timestamp como chave (dedup).
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from pipeline import model as mdl, positions, storage, telegram

DATA = Path("data")
SIGNALS = DATA / "signals.parquet"
STATE = DATA / "dashboard_state.json"


def _log_signal(pred: dict) -> int:
    """Append à signals.parquet (dedup por open_time)."""
    row = pl.DataFrame([{
        "open_time": pred["open_time"],
        "close": pred["close"],
        "proba_long": pred["proba_long"],
        "signal": int(pred["signal"]),
        "confidence_pct": pred["confidence_pct"],
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }])
    return storage.upsert(SIGNALS, row, "open_time")


def _load_state() -> dict | None:
    if not STATE.exists():
        return None
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return None


def run(quiet: bool = False, force_send: bool = False) -> None:
    print("[predict] treina dual-horizon (mid 48h + long 72h)…", flush=True)
    pred = mdl.predict_dual_horizon()
    mat = pred["_mat_mid"]  # pra acessar ATR atual

    ts = datetime.fromtimestamp(pred["open_time"] / 1000, tz=timezone.utc)
    print(
        f"[predict]   bar={ts:%Y-%m-%d %H:%M}  close=${pred['close']:,.0f}  "
        f"mid={pred['proba_mid']*100:.1f}%  long={pred['proba_long_horizon']*100:.1f}%  "
        f"signal={pred['signal']}  (mid={pred['signal_mid']}, long={pred['signal_long_h']})"
    )

    n = _log_signal(pred)
    print(f"[predict]   +{n} row em signals.parquet")

    has_signal = pred["signal"] or force_send
    if not has_signal:
        if not quiet:
            try:
                telegram.send(telegram.format_no_signal(pred), silent=True)
                print("[predict]   sem sinal — notificação silenciosa enviada")
            except Exception as e:
                print(f"[predict]   ⚠️ telegram silent falhou: {e}")
        else:
            print("[predict]   sem sinal — nada enviado (quiet mode)")
        return

    state = _load_state()
    is_test_mode = force_send and not pred["signal"]

    # Abre posição apenas em sinal REAL (não em teste forçado)
    new_pos = None
    if pred["signal"] and not is_test_mode:
        if positions.has_open():
            print("[predict] já existe posição aberta — não abre nova (evita duplicata)")
        else:
            atr_now = float(mat["atr_14"][-1]) if "atr_14" in mat.columns else None
            if atr_now and atr_now > 0:
                new_pos = positions.open_position(
                    entry_time_ms=pred["open_time"],
                    entry_price=pred["close"],
                    atr=atr_now,
                    proba_long=pred["proba_long"],
                    atr_mult=mdl.ATR_MULT,
                    horizon_hours=mdl.HORIZON_BARS * 4,  # bars 4h
                )
                # Sugestão de sizing — leverage dinâmica por padrão, env override pra manual.
                user_capital = float(os.environ.get("TELEGRAM_USER_CAPITAL", "1000"))
                rv_now = float(mat["rv_1d"][-1]) * (mdl.HORIZON_BARS * 365 / 12) ** 0.5 \
                    if "rv_1d" in mat.columns else None
                env_lev = os.environ.get("TRADING_LEVERAGE")
                if env_lev is not None:
                    lev_info = {"leverage": float(env_lev), "f_conf": None, "f_vol": None, "suppressed": False, "source": "env"}
                elif mdl.LEVERAGE_DYNAMIC:
                    lev_info = mdl.dynamic_leverage(
                        proba_mid=pred["proba_mid"],
                        rv_30d_ann=rv_now,
                        in_bear=pred["in_bear"],
                    )
                    lev_info["source"] = "dynamic"
                else:
                    lev_info = {"leverage": mdl.LEVERAGE_DEFAULT, "source": "default"}
                sz = mdl.position_size(
                    capital=user_capital,
                    entry_price=new_pos["entry_price"],
                    stop_price=new_pos["stop_price"],
                    leverage=lev_info["leverage"],
                )
                sz["capital"] = user_capital
                sz["leverage_info"] = lev_info
                new_pos["size_suggestion"] = sz

                print(
                    f"[predict] 📌 Posição aberta: entry=${new_pos['entry_price']:,.0f}  "
                    f"target=${new_pos['target_price']:,.0f}  stop=${new_pos['stop_price']:,.0f}  "
                    f"size_sugerido={sz['size_btc']:.5f} BTC (${sz['size_usd']:,.0f}, {sz['pct_of_capital']*100:.0f}% do capital)"
                )
            else:
                print("[predict] ⚠️ ATR inválido, não abriu posição")
    elif is_test_mode:
        # Em teste, mostra sizing como se fosse signal real (cosmético)
        atr_now = float(mat["atr_14"][-1]) if "atr_14" in mat.columns else None
        if atr_now and atr_now > 0:
            entry = pred["close"]
            stop = entry - mdl.ATR_MULT * atr_now
            target = entry + mdl.ATR_MULT * atr_now
            user_capital = float(os.environ.get("TELEGRAM_USER_CAPITAL", "1000"))
            rv_now = float(mat["rv_1d"][-1]) * (mdl.HORIZON_BARS * 365 / 12) ** 0.5 \
                if "rv_1d" in mat.columns else None
            env_lev = os.environ.get("TRADING_LEVERAGE")
            if env_lev is not None:
                lev_info = {"leverage": float(env_lev), "source": "env"}
            elif mdl.LEVERAGE_DYNAMIC:
                lev_info = mdl.dynamic_leverage(
                    proba_mid=pred["proba_mid"],
                    rv_30d_ann=rv_now,
                    in_bear=pred["in_bear"],
                )
                lev_info["source"] = "dynamic"
            else:
                lev_info = {"leverage": mdl.LEVERAGE_DEFAULT, "source": "default"}
            sz = mdl.position_size(capital=user_capital, entry_price=entry, stop_price=stop,
                                   leverage=lev_info["leverage"])
            sz["capital"] = user_capital
            sz["leverage_info"] = lev_info
            new_pos = {
                "entry_price": entry,
                "target_price": target,
                "stop_price": stop,
                "horizon_hours": mdl.HORIZON_BARS * 4,
                "size_suggestion": sz,
            }

    msg = telegram.format_signal(pred, state=state, is_test=is_test_mode, position=new_pos)
    try:
        telegram.send(msg)
        print(f"[predict] ✅ Alerta enviado")
    except Exception as e:
        print(f"[predict] ❌ Telegram falhou: {e}")
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quiet", action="store_true", help="Não envia notificação silenciosa quando sem sinal")
    p.add_argument("--force-send", action="store_true", help="Envia alerta mesmo sem sinal (teste)")
    run(**vars(p.parse_args()))
