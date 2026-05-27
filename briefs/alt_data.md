# Alt-data — diagnóstico e roadmap de integração

Projeto: btc-forecast (intraday 15m, LightGBM walk-forward, Sharpe 1.29 dual-horizon AND ensemble).

---

## Diagnóstico (alt-data gaps do projeto)

Stack atual de features:
- **Técnico** (preço/vol): ret_*, ATR, RSI multi-TF, MAs/dist, BB, Z-vol.
- **Derivativos**: funding rate Binance USDS-M apenas (`funding`, `funding_z90`, `funding_ema8d`, `hours_since_funding`).
- **Macro**: DXY/VIX/SPX (yfinance), shift dia D+1 06:00 UTC.
- **Sentiment**: F&G alternative.me + FinBERT em news (GDELT/CoinDesk).

Gaps confirmados — toda categoria abaixo é ZERO no pipeline:

1. **On-chain** — zero. Nenhuma feature derivada de UTXO/mempool/exchange wallets/miner flow. Para um sinal intraday 15m em BTC, ignorar o on-chain é o maior gap. Mempool fee pressure, exchange netflow e miner outflow são leading indicators documentados de regimes de stress (sell-side liquidity surge) e capitulation.
2. **Options (Deribit)** — zero. DVOL é o "VIX do BTC" e antecipa expansões de vol realizada (cf. Deribit Insights). Gamma exposure (GEX) explica price pinning em strikes redondos próximos a expiry mensal/quarterly. Put/call ratio é proxy direta de skew direcional. Tudo grátis via Deribit public JSON-RPC, sem auth.
3. **Liquidations / OI / long-short ratio** — zero. Funding sozinho não captura squeeze risk; OI + liquidations agregados explicam reversões violentas (cascade liquidations 1h). Hoje o modelo "vê" pressão de carry (funding) mas não "vê" alavancagem absoluta nem onde ela quebra.
4. **ETF spot flows (US)** — zero. Desde jan/2024 IBIT/FBTC/ARKB movem ~$200M-$500M/dia em fluxo direcional (Farside). Daily só, mas com lag D+1 ainda é forte (correlação contemporânea com retorno D+1 alta — ver Phemex academy / SoSoValue dashboards).
5. **Cross-leverage proxies** — zero. Open interest Binance (`fapi /futures/data/openInterestHist`) é endpoint público sem auth e o projeto não está usando. Coinalyze agrega OI/liq/LSR cross-venue grátis com API key — não está em uso.

Conclusão: o sinal hoje é técnico + macro + sentiment_news; falta toda a camada de **flow** (on-chain, derivativos além de funding, ETF). Hipótese: 30-50% das reversões/blow-offs que o modelo erra hoje são explicáveis por OI surge + liquidation cascade + DVOL spike, nenhum capturado.

Restrições do projeto:
- Sources daily (ETF, on-chain CryptoQuant grátis) entram via `available_at` lag D+1 — padrão `add_macro` já existe.
- Intraday (Deribit DVOL, mempool, Binance OI, Coinalyze) entram via `join_asof backward` no grid 15m — padrão `add_funding` já existe.
- Sem custo recorrente. Free tiers ou public endpoints só.

---

## Top-3 fontes a integrar primeiro

Critério: (a) custo zero, (b) granularidade ≥ 1h, (c) hipótese de lift sustentada por evidência publicada, (d) baixo atrito de engenharia.

### #1 — Deribit DVOL + put/call OI ratio (options)
- **Endpoint** (público, sem auth):
  - DVOL: `GET https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=BTC&start_timestamp=<ms>&end_timestamp=<ms>&resolution=60`
  - Options book summary (para PCR): `GET https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option` — retorna `open_interest` por `instrument_name` (parse `-P`/`-C` no nome).
- **Custo**: $0. JSON-RPC 2.0, rate limit generoso (público).
- **Granularidade**: DVOL — resolução 60s/1m em diante; histórico desde 2021. PCR — snapshot atual (precisa cron 15m gravando).
- **Lift esperado**: DVOL antecipa expansão de RV realizada com 0.5-2h de antecedência (cf. Deribit Insights, "DVOL — Deribit Implied Volatility Index"). Em LightGBM, esperar ganho de Sharpe +0.05-0.15 vinda principalmente de filtrar trades em regimes de vol crescente (false breakouts caem). PCR < 0.4 historicamente coincidiu com tops locais (cf. Glassnode Insights "Taker-Flow-Based Gamma Exposure", dez/2025 expiry $24B).
- **Evidência**: Deribit Insights DVOL paper; Glassnode gamma exposure article; CoinDesk dec 2025 BTC range $85-90k driven by gamma pinning.
- **Features derivadas**:
  - `dvol_15m` (nivel), `dvol_z30` (rolling 30d), `dvol_chg_4h` (delta short)
  - `pcr_oi` (put OI / call OI agregado), `pcr_oi_z14`
  - `dvol_minus_rv1d` (IV-RV spread — premium em opções vs vol realizada)

### #2 — Binance Open Interest histórico (15m perp)
- **Endpoint** (público, sem auth — mesmo host já usado pra funding):
  - `GET https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=15m&limit=500`
  - Cuidado: rate limit IP 1000 req / 5min. **Histórico só últimos 30 dias** — backfill incremental obrigatório (gravar continuamente via GH Actions cron 15m, igual ao OHLCV).
- **Custo**: $0.
- **Granularidade**: nativa 15m — match perfeito com o grid do modelo. Sem necessidade de as-of join, basta `join` em `open_time`.
- **Lift esperado**: OI Δ % vs price Δ % é o feature derivativo clássico (cf. ROADMAP seção 4 — "OI Δ vs preço Δ"). Combinado com funding já existente, dá leitura completa: funding (carry) + OI (positioning size) + price (direção). Esperar Sharpe +0.05-0.10. Esse era o item planejado e ainda não feito.
- **Evidência**: Binance dev community thread "Futers Open Interest Historical Data BTCUSDT"; Gate.io article sobre interpretação de funding+OI+liq.
- **Features derivadas**:
  - `oi_usd`, `oi_z30`, `oi_chg_1h`, `oi_chg_4h`
  - `oi_price_divergence` = sign(`oi_chg_4h`) ≠ sign(`ret_4h`) → contraria positioning
  - `ix_oi_funding` = `oi_z30` × `funding_z90` (squeeze risk composto)

### #3 — Coinalyze aggregated liquidations + long/short ratio
- **Endpoint** (free, API key — signup gratuito):
  - Base: `https://api.coinalyze.net/v1/`
  - `liquidation-history?symbols=BTCUSDT_PERP.A&interval=1hour&from=<ts>&to=<ts>` (`.A` = aggregated cross-venue)
  - `long-short-ratio-history?symbols=BTCUSDT_PERP.A&interval=1hour&...`
  - `open-interest-history?...` (alternativa cross-venue ao OI Binance-only)
- **Custo**: $0. Rate limit 40 req/min/key.
- **Granularidade**: 1min/5min/15min/30min/1h disponíveis. Histórico amplo (>1 ano, depende do par).
- **Lift esperado**: Liquidation cascades 1h são leading indicator de mean reversion violenta — modelo hoje pega o move só pós-fato via `rv_4h`. LSR (long-short ratio) é proxy direta de sentimento posicional retail (Binance top trader account ratio agregado). Esperar Sharpe +0.05-0.10 em regime bear/chop.
- **Evidência**: Coinalyze docs `api.coinalyze.net/v1/doc/`; Gate.io "How to interpret crypto derivatives market signals: funding rates, open interest, and liquidation data explained".
- **Features derivadas**:
  - `liq_long_usd_1h`, `liq_short_usd_1h`, `liq_imbalance_1h = (short-long)/(short+long)`
  - `liq_z30` (vol de stress geral)
  - `lsr_account`, `lsr_z14`

---

## Experimento concreto — MVP: integrar Binance OI (fonte #2)

Razão para começar por OI: (a) mesmo host que `binance.py` já usa, zero atrito de auth/key, (b) granularidade 15m nativa elimina as-of join, (c) era item ROADMAP fase 4 planejado e não feito, (d) baseline mais fácil pra medir lift incremental antes de pagar complexidade de Deribit/Coinalyze.

### 1. Backfill
Binance só serve ~30d históricos. Estratégia:
- Iniciar gravação contínua **agora** via novo workflow `ingest_oi_15m.yml` (cron */15 min, idêntico ao `ingest_15m.yml`).
- Para histórico extenso (>30d), usar `data.binance.vision` (dump diários estilo OHLCV) — alternativa de pesquisa adicional necessária.
- Treino inicial: rodar com 30d de OI + dados antigos sem OI → validar lift apenas no slice recente. Walk-forward só nos folds com OI presente.

### 2. Fetcher (proposals/altdata_fetchers.py — método `fetch_binance_oi_15m`)
- Loop paginado igual a `fetch_klines`, batch 500, sleep 0.25s.
- Persistir em `data/oi_15m.parquet` com schema `{open_time: i64, oi_usd: f64, oi_coin: f64}`.
- Aplicar mesmo `CLOSED_BUFFER_MS` que OHLCV (vela em formação fora).

### 3. Feature integration (pipeline/features.py — novo `add_oi`)
```python
def add_oi(df: pl.DataFrame, oi: pl.DataFrame) -> pl.DataFrame:
    o = oi.sort("open_time").with_columns(
        pl.col("oi_usd").alias("oi"),
        _rolling_zscore("oi_usd", BARS_PER_DAY * 30, "oi_z30"),
        pl.col("oi_usd").pct_change(BARS_PER_HOUR).alias("oi_chg_1h"),
        pl.col("oi_usd").pct_change(BARS_PER_HOUR * 4).alias("oi_chg_4h"),
    )
    return df.sort("open_time").join(
        o.select(["open_time","oi","oi_z30","oi_chg_1h","oi_chg_4h"]),
        on="open_time", how="left",
    )
```
- Plugar em `build_v2` após `add_funding`. `apply_lag` já cobre shift(1) automático.
- Adicionar interação `ix_oi_funding = oi_z30 * funding_z90` em `add_interactions`.

### 4. Validação (walk-forward)
- Refazer fold mensal de `exp_backtest_1k.py` apenas nos meses onde OI está presente.
- Métrica de aceite: Sharpe líquido fold-by-fold ≥ baseline +0.05 com profit factor não caindo. Senão, kill da feature.
- SHAP rank: `oi_z30` e `oi_chg_4h` precisam aparecer no top-20 com sinal consistente cross-fold. Se inconsistente, é noise.

### 5. Gate de promoção
- Se passar: estender pra fonte #1 (DVOL) com mesma estrutura.
- Se falhar: investigar 30d de histórico não bastam, ou OI Binance-only é viés (Coinalyze cross-venue é o fix).

---

## Refs

APIs (URLs verificadas):
- Deribit `public/get_volatility_index_data` — https://docs.deribit.com/api-reference/upcoming/market-data/public-get_volatility_index_data
- Deribit `public/get_book_summary_by_currency` — https://docs.deribit.com/api-reference/market-data/public-get_book_summary_by_currency
- Binance USDS-M Futures Open Interest Statistics — https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics
- Coinalyze API doc — https://api.coinalyze.net/v1/doc/
- CoinGlass aggregated liquidation history — https://docs.coinglass.com/reference/aggregated-liquidation-history (free tier limitado; backup pago)
- mempool.space REST — https://mempool.space/docs/api/rest
- Blockchain.com Charts API — https://www.blockchain.com/api/charts_api (hash-rate, mempool-size, miners-revenue, n-transactions — JSON/CSV, free)
- CryptoQuant API user guide (paid) — https://userguide.cryptoquant.com/api/btc-exchange-flows
- Farside BTC ETF flows table — https://farside.co.uk/btc/ (HTML scrape; CSV não público)
- SoSoValue ETF dashboard — https://sosovalue.com/assets/etf/us-btc-spot
- Glassnode Studio (US Spot ETF Flows Net chart, public) — https://studio.glassnode.com/charts/institutions.UsSpotEtfFlowsNet?a=BTC

Insights / context:
- Deribit Insights — "DVOL: Deribit Implied Volatility Index" — https://insights.deribit.com/exchange-updates/dvol-deribit-implied-volatility-index/
- Glassnode Insights — "Taker-Flow-Based Gamma Exposure" — https://insights.glassnode.com/gamma-exposure/

Papers:
- Liu, Tsyvinski (2021) — "Risks and Returns of Cryptocurrency", Review of Financial Studies — https://academic.oup.com/rfs/article/34/6/2689/5912024
- "Return and Volatility Forecasting Using On-Chain Flows in Cryptocurrency Markets" (2024, arXiv) — https://arxiv.org/pdf/2411.06327 — relevante: USDT exchange inflows preveem retorno BTC/ETH intraday 1-6h.
- "Bitcoin volatility in bull vs bear market — on-chain metrics + Twitter" — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10773860/
- "Turn-of-the-candle effect in bitcoin returns" — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10015199/ (calendar effect 15min, ortogonal mas relevante pro grid 15m do projeto)
