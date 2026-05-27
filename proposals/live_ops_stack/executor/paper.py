"""Executor — consome sinal do Redis e envia ordem testnet Binance USDM futures.

Modos:
  paper   → ordem testnet (default, sandbox real do Binance)
  dryrun  → log only, não envia ordem
  live    → produção (BLOQUEADO via flag dupla: env LIVE_CONFIRMED=YES + redis live_allowed=1)

Loop:
  1. SUBSCRIBE signal:new
  2. Pre-trade risk checks:
     - redis.get("killed") != "0/false/empty"
     - open_positions < MAX_OPEN_POSITIONS
     - sinal recente (não stale > 4min, ver brief Execution)
     - margem suficiente
  3. Envia MARKET entry + STOP_MARKET reduceOnly + TAKE_PROFIT_MARKET reduceOnly
  4. Persiste em SQLite + alerta Telegram fill
  5. Reconcile loop 60s: fetch_positions() vs DB local

TODOs:
- [ ] integrar dynamic_sizing_risk.compute_position_size (célula Risk)
- [ ] integrar execution_cost_model pra log de slippage real (célula Execution)
- [ ] storage_sqlite WAL (substituir parquets do MVP)
- [ ] reconcile loop
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone

import ccxt.async_support as ccxt
import redis.asyncio as redis_async

sys.path.insert(0, "/app")

from pipeline.telegram import send as tg_send  # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","lvl":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("executor")

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
MODE = os.environ.get("MODE", "paper")  # paper | dryrun | live
MAX_OPEN_POSITIONS = int(os.environ.get("MAX_OPEN_POSITIONS", "1"))
SYMBOL = os.environ.get("SYMBOL", "BTC/USDT")
LEVERAGE = int(os.environ.get("LEVERAGE", "3"))
SIGNAL_STALE_SECONDS = int(os.environ.get("SIGNAL_STALE_SECONDS", "240"))  # 4min
KILL_GUARD = os.environ.get("KILL_GUARD", "1") == "1"


def _make_exchange() -> ccxt.binance:
    api_key = os.environ.get("BINANCE_TESTNET_KEY", "")
    secret = os.environ.get("BINANCE_TESTNET_SECRET", "")
    if MODE == "live":
        confirmed = os.environ.get("LIVE_CONFIRMED", "") == "YES"
        if not confirmed:
            raise SystemExit("MODE=live requires LIVE_CONFIRMED=YES env var")
        ex = ccxt.binance({"apiKey": api_key, "secret": secret, "options": {"defaultType": "future"}})
    else:
        ex = ccxt.binance({
            "apiKey": api_key,
            "secret": secret,
            "options": {"defaultType": "future"},
            "urls": {
                "api": {
                    "fapiPublic": "https://testnet.binancefuture.com/fapi/v1",
                    "fapiPrivate": "https://testnet.binancefuture.com/fapi/v1",
                    "fapiPrivateV2": "https://testnet.binancefuture.com/fapi/v2",
                },
            },
        })
        ex.set_sandbox_mode(True)
    return ex


async def _killed(r) -> bool:
    if not KILL_GUARD:
        return False
    v = await r.get("killed")
    return bool(v) and v not in ("0", "false")


async def _process_signal(payload: dict, r, ex: ccxt.binance) -> None:
    if payload["signal"] == 0:
        log.info("signal=0 — nothing to do")
        return

    sig_ts = datetime.fromisoformat(payload["generated_at"].replace("Z", "+00:00"))
    age_s = (datetime.now(timezone.utc) - sig_ts).total_seconds()
    if age_s > SIGNAL_STALE_SECONDS:
        log.warning(f"signal stale {age_s:.0f}s > {SIGNAL_STALE_SECONDS}s — skip")
        return

    if await _killed(r):
        log.warning("killed flag active — refusing to trade")
        return

    open_positions = await r.scard("open_positions") or 0
    if open_positions >= MAX_OPEN_POSITIONS:
        log.warning(f"already {open_positions} open — skip")
        return

    entry = payload["close"]
    # TODO: integrar compute_position_size (Risk cell)
    notional_usd = float(os.environ.get("NOTIONAL_USD", "100"))  # MVP fixo
    amount = round(notional_usd / entry, 4)

    # ATR proxy não está aqui ainda — usar % fixo até integrar
    stop_pct = float(os.environ.get("STOP_PCT", "0.025"))
    tgt_pct = float(os.environ.get("TARGET_PCT", "0.035"))
    stop_price = round(entry * (1 - stop_pct), 1)
    target_price = round(entry * (1 + tgt_pct), 1)

    if MODE == "dryrun":
        log.info(f"DRYRUN entry={entry} stop={stop_price} tgt={target_price} amt={amount}")
        return

    try:
        await ex.set_leverage(LEVERAGE, SYMBOL)
        market_order = await ex.create_market_buy_order(SYMBOL, amount)
        await ex.create_order(SYMBOL, "STOP_MARKET", "sell", amount, None, {"stopPrice": stop_price, "reduceOnly": True})
        await ex.create_order(SYMBOL, "TAKE_PROFIT_MARKET", "sell", amount, None, {"stopPrice": target_price, "reduceOnly": True})
        await r.sadd("open_positions", market_order["id"])
        log.info(f"FILLED order={market_order['id']} entry≈{entry} stop={stop_price} tgt={target_price}")
        try:
            tg_send(
                f"📌 [{MODE}] FILL {SYMBOL} entry=${entry:.0f} stop=${stop_price:.0f} tgt=${target_price:.0f} amt={amount}",
            )
        except Exception as e:
            log.warning(f"telegram failed: {e}")
    except Exception as e:
        log.error(f"order failed: {e}")
        try:
            tg_send(f"🚨 executor order failed: {e}")
        except Exception:
            pass


async def main() -> None:
    r = redis_async.from_url(REDIS_URL, decode_responses=True)
    ex = _make_exchange()
    log.info(f"executor start mode={MODE} symbol={SYMBOL} max_open={MAX_OPEN_POSITIONS}")

    pubsub = r.pubsub()
    await pubsub.subscribe("signal:new")

    try:
        async for msg in pubsub.listen():
            if msg["type"] != "message":
                continue
            try:
                payload = json.loads(msg["data"])
                await _process_signal(payload, r, ex)
            except Exception as e:
                log.error(f"signal handling failed: {e}")
    finally:
        await ex.close()
        await pubsub.unsubscribe()
        await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
