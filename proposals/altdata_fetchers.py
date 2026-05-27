"""Stub de fetchers alt-data — proposta de integração.

NÃO está em uso. Segue o padrão de `pipeline/binance.py`: requests síncrono,
paginação com cursor, sleep entre batches, filtro de vela em formação.

Endpoints documentados em `/briefs/alt_data.md`.

Fontes priorizadas:
  1. Deribit DVOL + options book summary (público, sem auth)
  2. Binance USDS-M open interest histórico (público, sem auth — só ~30d)
  3. Coinalyze aggregated liquidations / long-short ratio (free, API key)
  4. Farside / Blockchain.com Charts (free; ETF flows, on-chain proxies)

Cada método retorna polars.DataFrame com schema explícito no docstring.
Persistência fica fora desse módulo — caller grava em data/*.parquet.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import polars as pl
import requests


# --------------------------------------------------------------------- consts
DERIBIT_BASE = "https://www.deribit.com/api/v2"
BINANCE_FAPI = "https://fapi.binance.com"
COINALYZE_BASE = "https://api.coinalyze.net/v1"
BLOCKCHAIN_CHARTS = "https://api.blockchain.info/charts"
FARSIDE_BTC_ETF = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
MEMPOOL_BASE = "https://mempool.space/api"

SYMBOL = "BTCUSDT"
CLOSED_BUFFER_MS = 60 * 1000


def _now_ms() -> int:
    return int(time.time() * 1000)


# ========================================================== AltDataFetcher ==
@dataclass
class AltDataFetcher:
    """Coletor unificado de alt-data.

    Uso esperado (após implementação):
        f = AltDataFetcher(coinalyze_api_key=os.getenv("COINALYZE_API_KEY"))
        dvol = f.fetch_deribit_dvol(start_ms=..., resolution="60")  # 1min
        oi   = f.fetch_binance_oi_15m(start_ms=...)
        liq  = f.fetch_liquidations(interval="1hour", from_ts=..., to_ts=...)
        flow = f.fetch_etf_flows()
    """

    coinalyze_api_key: str | None = None
    timeout: int = 20
    sleep_between: float = 0.25

    # ---------------------------------------------------------- Deribit DVOL
    def fetch_deribit_dvol(
        self,
        start_ms: int,
        end_ms: int | None = None,
        resolution: str = "60",  # "1", "60", "3600", "43200", "1D"
    ) -> pl.DataFrame:
        """Deribit Implied Volatility Index (BTC).

        Endpoint: GET /public/get_volatility_index_data
        Params:
          currency=BTC, start_timestamp, end_timestamp (ms), resolution (segundos)
        Public, no auth.

        Returns schema:
          { open_time: i64, dvol_open: f64, dvol_high: f64,
            dvol_low: f64, dvol_close: f64 }

        TODO:
          - Loop paginado: Deribit limita ~5000 pontos por chamada.
            Se end-start / resolution > 5000, fatiar em chunks.
          - Tratar response shape: {"result": {"data": [[ts,o,h,l,c], ...]}}
          - Filtrar últimas N velas em formação (resolution >= 60s).
        """
        end_ms = end_ms or _now_ms()
        url = f"{DERIBIT_BASE}/public/get_volatility_index_data"
        params: dict[str, Any] = {
            "currency": "BTC",
            "start_timestamp": start_ms,
            "end_timestamp": end_ms,
            "resolution": resolution,
        }
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json().get("result", {}).get("data", [])
        if not data:
            return pl.DataFrame()
        # Cada row: [timestamp_ms, open, high, low, close]
        return pl.DataFrame(
            [
                {
                    "open_time": int(row[0]),
                    "dvol_open": float(row[1]),
                    "dvol_high": float(row[2]),
                    "dvol_low": float(row[3]),
                    "dvol_close": float(row[4]),
                }
                for row in data
            ]
        ).sort("open_time")

    # ----------------------------------------------- Deribit options PCR (OI)
    def fetch_deribit_pcr_snapshot(self) -> pl.DataFrame:
        """Snapshot atual: put/call open interest ratio para BTC options.

        Endpoint: GET /public/get_book_summary_by_currency
        Params: currency=BTC, kind=option
        Public, no auth.

        Retorna 1 linha:
          { snapshot_ms: i64, oi_calls: f64, oi_puts: f64, pcr: f64,
            n_calls: i32, n_puts: i32 }

        TODO:
          - Parse `instrument_name` no padrão "BTC-{date}-{strike}-{C|P}".
          - Pra histórico de PCR, gravar snapshot a cada 15m via cron
            (Deribit não expõe PCR histórico direto via API pública).
          - Considerar `get_instruments` + `ticker` se quiser GEX completo
            (gamma weighted por strike-spot distance).
        """
        url = f"{DERIBIT_BASE}/public/get_book_summary_by_currency"
        params = {"currency": "BTC", "kind": "option"}
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        rows = r.json().get("result", [])
        if not rows:
            return pl.DataFrame()

        oi_calls = oi_puts = 0.0
        n_calls = n_puts = 0
        for row in rows:
            name = row.get("instrument_name", "")
            oi = float(row.get("open_interest", 0.0) or 0.0)
            if name.endswith("-C"):
                oi_calls += oi
                n_calls += 1
            elif name.endswith("-P"):
                oi_puts += oi
                n_puts += 1

        pcr = oi_puts / oi_calls if oi_calls > 0 else None
        return pl.DataFrame(
            [
                {
                    "snapshot_ms": _now_ms(),
                    "oi_calls": oi_calls,
                    "oi_puts": oi_puts,
                    "pcr": pcr,
                    "n_calls": n_calls,
                    "n_puts": n_puts,
                }
            ]
        )

    # ------------------------------------------------ Binance USDS-M open interest
    def fetch_binance_oi_15m(
        self,
        start_ms: int,
        end_ms: int | None = None,
    ) -> pl.DataFrame:
        """Open interest histórico 15m do perp BTCUSDT (Binance USDS-M).

        Endpoint: GET /futures/data/openInterestHist
        Params: symbol=BTCUSDT, period=15m, startTime, endTime, limit=500
        Public, no auth. Rate limit 1000 req / 5 min / IP.

        WARNING: histórico limitado a ~30 dias. Backfill profundo exige
        data.binance.vision dumps (não coberto aqui — TODO separado).

        Returns schema:
          { open_time: i64, oi_coin: f64, oi_usd: f64 }
        """
        end_ms = end_ms or _now_ms()
        url = f"{BINANCE_FAPI}/futures/data/openInterestHist"
        rows: list[dict] = []
        cursor = start_ms
        while cursor < end_ms:
            r = requests.get(
                url,
                params={
                    "symbol": SYMBOL,
                    "period": "15m",
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 500,
                },
                timeout=self.timeout,
            )
            if r.status_code == 451:
                # Mesmo geo-block que /fapi/v1/fundingRate sofre em alguns runners.
                print("[oi] WARN: 451 geo-block, pulando coleta")
                return pl.DataFrame()
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows.extend(batch)
            cursor = int(batch[-1]["timestamp"]) + 1
            time.sleep(self.sleep_between)
            if len(batch) < 500:
                break

        if not rows:
            return pl.DataFrame()

        cutoff = _now_ms() - CLOSED_BUFFER_MS
        df = pl.DataFrame(
            [
                {
                    "open_time": int(r["timestamp"]),
                    "oi_coin": float(r["sumOpenInterest"]),
                    "oi_usd": float(r["sumOpenInterestValue"]),
                }
                for r in rows
            ]
        )
        return df.filter(pl.col("open_time") <= cutoff).sort("open_time")

    # ------------------------------------------------ Coinalyze liquidations
    def fetch_liquidations(
        self,
        symbols: str = "BTCUSDT_PERP.A",  # .A = aggregated cross-venue
        interval: str = "1hour",  # 1min, 5min, 15min, 30min, 1hour, 4hour, 1day
        from_ts: int | None = None,
        to_ts: int | None = None,
        convert_to_usd: bool = True,
    ) -> pl.DataFrame:
        """Aggregated liquidation history (Coinalyze).

        Endpoint: GET /v1/liquidation-history
        Headers: api_key: <key>
        Rate limit: 40 req/min/key. Free signup.

        Returns schema:
          { open_time: i64, liq_long_usd: f64, liq_short_usd: f64,
            liq_total_usd: f64 }

        TODO:
          - Confirmar shape exato da resposta no schema oficial. Doc atualmente
            inacessível via WebFetch — testar manualmente e ajustar parse.
          - Coinalyze devolve `t` em SEGUNDOS (não ms) — converter.
          - Paginação: se range > limite, fatiar.
        """
        if not self.coinalyze_api_key:
            raise RuntimeError(
                "COINALYZE_API_KEY missing — signup em https://coinalyze.net/"
            )

        url = f"{COINALYZE_BASE}/liquidation-history"
        params = {
            "symbols": symbols,
            "interval": interval,
            "convert_to_usd": str(convert_to_usd).lower(),
        }
        if from_ts is not None:
            params["from"] = from_ts // 1000  # API espera segundos
        if to_ts is not None:
            params["to"] = to_ts // 1000

        r = requests.get(
            url,
            params=params,
            headers={"api_key": self.coinalyze_api_key},
            timeout=self.timeout,
        )
        r.raise_for_status()
        payload = r.json()
        # TODO: confirmar shape — geralmente [{ "symbol": ..., "history": [...] }]
        if not payload:
            return pl.DataFrame()
        history = payload[0].get("history", []) if isinstance(payload, list) else []
        if not history:
            return pl.DataFrame()

        return pl.DataFrame(
            [
                {
                    "open_time": int(row["t"]) * 1000,  # s -> ms
                    "liq_long_usd": float(row.get("l", 0.0) or 0.0),
                    "liq_short_usd": float(row.get("s", 0.0) or 0.0),
                    "liq_total_usd": float(row.get("l", 0.0) or 0.0)
                    + float(row.get("s", 0.0) or 0.0),
                }
                for row in history
            ]
        ).sort("open_time")

    # ---------------------------- Coinalyze long-short ratio (mesmo padrão acima)
    def fetch_long_short_ratio(
        self,
        symbols: str = "BTCUSDT_PERP.A",
        interval: str = "1hour",
        from_ts: int | None = None,
        to_ts: int | None = None,
    ) -> pl.DataFrame:
        """Endpoint: GET /v1/long-short-ratio-history. Mesma auth do liq.

        Returns schema:
          { open_time: i64, ls_ratio: f64, longs_pct: f64, shorts_pct: f64 }

        TODO: validar nomes de campo no payload real.
        """
        raise NotImplementedError("TODO — espelhar fetch_liquidations")

    # ----------------------------------------- Exchange netflow proxy (free)
    def fetch_exchange_netflow_proxy(
        self,
        timespan: str = "1year",
    ) -> pl.DataFrame:
        """Proxy gratuito on-chain via Blockchain.com Charts API.

        Não é netflow real (precisa CryptoQuant/Glassnode paid),
        mas combina:
          - n-transactions (tx count diário) — proxy de atividade
          - transaction-fees-usd — proxy de pressão de blockspace
          - miners-revenue — proxy de selling pressure miners
          - hash-rate — proxy de capitulation/expansion

        Endpoint: GET https://api.blockchain.info/charts/<chart_name>
        Params: timespan=1year, format=json
        Public, no auth.

        Returns schema (long format, daily):
          { date: Date, metric: str, value: f64 }

        TODO:
          - Para netflow REAL, integrar:
            (a) CryptoQuant Free dashboards (HTML scrape — frágil) OU
            (b) Glassnode Studio public charts (HTML) OU
            (c) bitcoin-data.com / coinmetrics community node OU
            (d) self-hosted electrum + script de wallet labeling (caro).
          - available_at = dia D+1 (lag pra evitar look-ahead, padrão `add_macro`).
        """
        charts = ["n-transactions", "transaction-fees-usd", "miners-revenue", "hash-rate"]
        frames: list[pl.DataFrame] = []
        for chart in charts:
            r = requests.get(
                f"{BLOCKCHAIN_CHARTS}/{chart}",
                params={"timespan": timespan, "format": "json"},
                timeout=self.timeout,
            )
            r.raise_for_status()
            values = r.json().get("values", [])
            if not values:
                continue
            frames.append(
                pl.DataFrame(
                    [
                        {
                            "date": int(v["x"]) * 1000,  # unix s -> ms
                            "metric": chart,
                            "value": float(v["y"]),
                        }
                        for v in values
                    ]
                )
            )
            time.sleep(self.sleep_between)

        if not frames:
            return pl.DataFrame()
        return pl.concat(frames).sort(["metric", "date"])

    # ------------------------------------ Mempool pressure (intraday on-chain)
    def fetch_mempool_snapshot(self) -> pl.DataFrame:
        """Snapshot atual do mempool — gravar via cron 15m pra build histórico.

        Endpoints:
          GET /mempool                     — count, vsize, total_fee
          GET /v1/fees/recommended         — fee rates (sat/vB)
          GET /v1/fees/mempool-blocks      — projected blocks

        Public, no auth.

        Returns schema (1 linha):
          { snapshot_ms, mempool_count, mempool_vsize, mempool_total_fee,
            fee_fastest, fee_half_hour, fee_hour, fee_economy, fee_minimum }
        """
        out: dict[str, Any] = {"snapshot_ms": _now_ms()}

        r1 = requests.get(f"{MEMPOOL_BASE}/mempool", timeout=self.timeout)
        r1.raise_for_status()
        m = r1.json()
        out["mempool_count"] = int(m.get("count", 0))
        out["mempool_vsize"] = int(m.get("vsize", 0))
        out["mempool_total_fee"] = int(m.get("total_fee", 0))

        r2 = requests.get(f"{MEMPOOL_BASE}/v1/fees/recommended", timeout=self.timeout)
        r2.raise_for_status()
        f = r2.json()
        out["fee_fastest"] = int(f.get("fastestFee", 0))
        out["fee_half_hour"] = int(f.get("halfHourFee", 0))
        out["fee_hour"] = int(f.get("hourFee", 0))
        out["fee_economy"] = int(f.get("economyFee", 0))
        out["fee_minimum"] = int(f.get("minimumFee", 0))

        return pl.DataFrame([out])

    # ---------------------------------------------- ETF flows (Farside daily)
    def fetch_etf_flows(self) -> pl.DataFrame:
        """Bitcoin spot ETF daily flows (US$m) — Farside Investors.

        Source: https://farside.co.uk/bitcoin-etf-flow-all-data/
        Sem API pública. Estratégias possíveis:
          1. HTML scrape com pandas.read_html (a tabela é estática).
          2. Glassnode Studio public chart (institutions.UsSpotEtfFlowsNet) —
             requer parsing de XHR no Studio (mais frágil).
          3. SoSoValue API open platform — anúncio recente, requer signup.
             https://m.sosovalue.com/

        Returns schema:
          { date: Date, ibit_usd_m: f64, fbtc_usd_m: f64, bitb_usd_m: f64,
            arkb_usd_m: f64, btco_usd_m: f64, ezbc_usd_m: f64,
            brrr_usd_m: f64, hodl_usd_m: f64, btcw_usd_m: f64,
            gbtc_usd_m: f64, btc_usd_m: f64, total_net_usd_m: f64 }

        Status: NÃO IMPLEMENTADO. Farside retornou 403 a User-Agents
        automatizados em teste — provavelmente exige headers de browser
        (Mozilla, Accept, etc.) + retry. Alternativa robusta: SoSoValue API
        após signup.

        TODO:
          - Tentar `requests.get(FARSIDE_BTC_ETF, headers={"User-Agent": "..."})`
            e parsear com `pd.read_html` -> primeira tabela com colunas
            ['Date', 'IBIT', 'FBTC', ...].
          - Skip headers como "Total", subtotais "1st week April", etc.
          - Parse números formato "(123.4)" como -123.4 (parênteses = negativo).
          - available_at = date + 1 dia (lag igual `add_macro`).
        """
        raise NotImplementedError(
            "TODO — Farside scrape requer headers de browser; "
            "ou usar SoSoValue API após signup."
        )


# ============================================================ smoke (manual)
if __name__ == "__main__":
    # Exemplo de uso (não rodar em CI até implementação validada).
    fetcher = AltDataFetcher(
        coinalyze_api_key=os.getenv("COINALYZE_API_KEY"),
    )
    # 24h de DVOL em resolução 1h
    end = _now_ms()
    start = end - 24 * 60 * 60 * 1000
    dvol = fetcher.fetch_deribit_dvol(start_ms=start, end_ms=end, resolution="3600")
    print("DVOL rows:", len(dvol))
    print(dvol.head())
