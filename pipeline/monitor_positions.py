"""Job que monitora posições abertas e fecha quando atinge target/stop/timeout.

Executado pelo workflow monitor_positions.yml (cron a cada 15min).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from pipeline import positions, storage, telegram

OHLCV = Path("data") / "ohlcv_15m.parquet"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _format_exit(pos: dict, new_state: dict) -> str:
    """Mensagem Telegram pra saída."""
    status = new_state["status"]
    exit_dt = datetime.fromtimestamp(new_state["exit_time"] / 1000, tz=timezone.utc)
    entry_dt = datetime.fromtimestamp(pos["entry_time"] / 1000, tz=timezone.utc)
    pnl_raw = new_state["exit_price"] / pos["entry_price"] - 1
    pnl_net = pnl_raw - positions.COST_ROUND
    duration = (new_state["exit_time"] - pos["entry_time"]) / 3600000  # horas

    emoji, header = {
        "closed_target": ("🟢", "*ALVO ATINGIDO — saída lucro*"),
        "closed_stop": ("🔴", "*STOP ATINGIDO — saída perda*"),
        "closed_timeout": ("⏱️", "*TIMEOUT — saída por tempo*"),
    }[status]

    lines = [
        f"{emoji} {header}",
        "",
        f"📊 Entrada: `{entry_dt:%Y-%m-%d %H:%M} UTC` @ *${pos['entry_price']:,.0f}*",
        f"📊 Saída:   `{exit_dt:%Y-%m-%d %H:%M} UTC` @ *${new_state['exit_price']:,.0f}*",
        f"⏱️  Duração: {duration:.1f}h",
        "",
        f"💰 P&L bruto: *{pnl_raw*100:+.2f}%*",
        f"💰 P&L líquido (–0.08% cost): *{pnl_net*100:+.2f}%*",
    ]

    # Comparação ao paper
    if status == "closed_target":
        lines.append("\n✅ Modelo acertou — barreira superior batida.")
    elif status == "closed_stop":
        lines.append("\n❌ Modelo errou — barreira inferior batida.")
    else:
        lines.append("\n⚪ Sem direção clara em 48h — mercado lateral.")

    return "\n".join(lines)


def run() -> None:
    open_pos = positions.get_open()
    if open_pos.is_empty():
        print("[monitor] sem posições abertas")
        return

    print(f"[monitor] {open_pos.height} posições abertas")

    if not OHLCV.exists():
        print("[monitor] ⚠️ ohlcv_15m.parquet ausente, pulando")
        return

    ohlcv = pl.read_parquet(OHLCV)
    now_ms = _now_ms()
    closed_count = 0

    for pos in open_pos.to_dicts():
        new_state = positions.evaluate_position(pos, ohlcv, now_ms)
        if new_state is None:
            current_price = float(ohlcv["close"][-1])
            pnl_now = current_price / pos["entry_price"] - 1
            print(
                f"  open #{pos['entry_time']}  entry=${pos['entry_price']:,.0f}  "
                f"now=${current_price:,.0f}  pnl_now={pnl_now*100:+.2f}%  "
                f"target=${pos['target_price']:,.0f}  stop=${pos['stop_price']:,.0f}"
            )
            continue

        updated = positions.close_position(
            entry_time_ms=pos["entry_time"],
            exit_time_ms=new_state["exit_time"],
            exit_price=new_state["exit_price"],
            status=new_state["status"],
        )
        print(
            f"  CLOSED #{pos['entry_time']}  status={new_state['status']}  "
            f"pnl={updated['pnl_pct']*100:+.2f}%"
        )
        closed_count += 1
        try:
            telegram.send(_format_exit(pos, new_state))
        except Exception as e:
            print(f"  ⚠️ telegram falhou: {e}")

    if closed_count == 0:
        print("[monitor] nenhuma posição fechada nesta iteração")
    else:
        print(f"[monitor] {closed_count} posições fechadas")


if __name__ == "__main__":
    run()
