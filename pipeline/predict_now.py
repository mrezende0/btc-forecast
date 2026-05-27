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

from pipeline import model as mdl, storage, telegram

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
    print("[predict] build matriz…", flush=True)
    mat, fcols = mdl.build_training_matrix()
    print(f"[predict]   {mat.height} rows, {len(fcols)} features")

    print("[predict] treina…", flush=True)
    model = mdl.train(mat, fcols)

    print("[predict] prediz…", flush=True)
    pred = mdl.predict_latest(model, mat, fcols)
    ts = datetime.fromtimestamp(pred["open_time"] / 1000, tz=timezone.utc)
    print(
        f"[predict]   bar={ts:%Y-%m-%d %H:%M}  close=${pred['close']:,.0f}  "
        f"proba={pred['proba_long']*100:.1f}%  signal={pred['signal']}"
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
    msg = telegram.format_signal(pred, state=state, is_test=force_send and not pred["signal"])
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
