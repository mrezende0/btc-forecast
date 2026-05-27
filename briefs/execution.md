# execution.md — validação de custo, slippage, latência e exchange choice

> Escopo: btc-forecast intraday 15m / sinal 4h, LightGBM dual-horizon, capital alvo $1k → escalar. Custo hardcoded `COST_ROUND = 0.0008` (8 bps round-trip = 5 bps taker + 3 bps slippage). Entrada via `predict_now.py` no close[t] após cron `5 0,4,8,12,16,20 * * *`.

## Diagnóstico

**Estado atual.** `pipeline/positions.py:28` e `notebooks/exp_backtest_1k.py:40` aplicam custo fixo `0.0008` no PnL líquido. Não há:

- Modelo de impacto por size (constante regardless of notional).
- Diferenciação maker vs taker (assume sempre taker market).
- Diferenciação spot vs perp / Binance vs Bybit vs OKX / cash vs margin.
- Slippage condicional ao regime de vol (ATR já calculado mas não usado p/ slippage).
- Latência cron→fill modelada (assume entrada no close[t] exato, sem skid).
- Spread bid-ask histórico (book L1 nunca foi coletado — `pipeline/binance.py` só puxa OHLCV + funding).
- Análise de partial fill / queue position em maker.

**Avaliação do 8 bps round-trip @ $1k.**

| Componente | Hardcoded | Realista @ $1k BTCUSDT spot Binance 2026 | Realista @ $50k |
|---|---|---|---|
| Taker fee | 5 bps | 4.5 bps (VIP0 spot) ou 3.825 bps (com BNB-25%) | idem (size irrelevante p/ tier até $1M/30d) |
| Spread/2 (half-spread no agressor) | (incluído) | 0.3–1.5 bps em horário líquido; 3–8 bps em US close / FOMC / liquidações | idem (book BTCUSDT top-of-book ≈ $10–50 wide → 0.3–1.5 bps em $100k preço) |
| Impact (square-root, η√(σ²·X/V)) | 0 | ~0.05 bps ($1k / vol-diário ~$20B spot → X/V ≈ 5e-8) | ~0.3 bps |
| Slippage adverse selection | implícito | 0.5–2 bps em ordem aggressive durante move | 1–4 bps |
| **Round-trip total** | **8 bps** | **10–22 bps modo normal · 25–60 bps modo stress** | **11–25 bps · 30–80 bps stress** |

Verdade: **a 8 bps subestima sistematicamente o custo em ~30–50% em condições normais e em >3× durante eventos macro / FOMC / CPI / liquidations**. O Sharpe 1.29 reportado em `exp_backtest_1k.py` é high-water; estimativa pessimista descontando 5–10 bps adicionais de custo → Sharpe ~0.95–1.10. Ainda passa o critério mínimo 0.7 do ROADMAP, mas come a margem.

**Onde dói mais.** Sinal dispara em close[t] que coincide frequentemente com candle de reversão / breakout → entrada como taker no exato momento de menor liquidez (queue de stops e momentum chasers). Slippage realizada > slippage média histórica → bias estilo "Markov-modulated" Cartea-Jaimungal ch.7 (impact dependente do estado).

**Latência cron→fill.** Cron `5 0,4,8,12,16,20 * * *` dá 5min de folga após close. Empiricamente GH Actions hosted runners ficam em queue 30–120s + setup-python 20–40s + pip install ~40s + treino LightGBM dual horizon ~30–90s + predict ~2s → **fill efetivo ~5min 30s até ~9min após close[t]**. Sinal é "stale" por ~5 min em market 4h (= 2% do bar). Em mercado calmo: irrelevante. Em pump/dump: 5min @ vol realizada 4h pode ser ±20–60 bps drift puro contra/a favor da entrada.

## Top-3 gaps

### Gap 1 — Slippage constante ignora size E regime (impacto Sharpe realista −0.15 a −0.30)
**Modelagem atual:** 3 bps fixos.
**Modelagem correta (Almgren-Chriss 2000 + Tóth 2011 square-root):**

```
slippage_bp = half_spread_bp + η · σ_bar · sqrt(participation_rate) · 1e4
participation_rate = order_size_usd / (volume_bar_usd × execution_window_frac)
η ≈ 0.5–1.0 (calibrado empiricamente p/ BTC spot — Donier-Bouchaud 2015)
```

A $1k notional em BTCUSDT 4h ($300M+ volume/4h):
- participation_rate ≈ 3.3e-6
- σ_bar 4h ≈ 60 bps em vol normal
- impact ≈ 0.5 · 60 · √(3.3e-6) · 1e4 ≈ 0.005 bps — **desprezível**

Então a $1k o gap real **não é impact, é half-spread + adverse selection**. Mas escalando p/ $50k a 1M (target da fase 9), impact passa de 0 → 5–20 bps.

**Sharpe report → realista:**
- 1.29 → ~1.05 (apenas re-cotando custo médio 12 bps vs 8 bps).
- Se 30% dos trades caem em janela de stress (FOMC week, US close 16:00 UTC, liquidation cascade), o tail-cost adiciona −0.10 a −0.20.

### Gap 2 — Maker vs taker nunca avaliado (impacto Sharpe potencial +0.10 a +0.25)
Binance spot 2026 (VIP0):
- Taker: 0.045% (com BNB pay: 0.03825%)
- Maker: 0.045% (sem rebate em VIP0; a partir de VIP1 + BNB cai pra 0.027%)

Comparativos:
- **Binance Futures (BTCUSDT perp)** VIP0: maker 0.02%, taker 0.05% → maker-first economiza 3 bps vs spot mid.
- **Bybit perpetual VIP0**: maker 0.02%, taker 0.055%.
- **OKX perp Tier 1**: maker 0.02%, taker 0.05%.
- **Hyperliquid**: maker 0.015% (rebate em alguns tiers via vault), taker 0.045% — book mais fino, mas custo nominal mais baixo.

Estratégia maker-first com fallback taker (post-only @ best bid +1 tick, cancela após 30s e cruza):
- Fill rate empírico em BTC perp top-of-book ≈ 60–75% em janelas 30–60s (Aquilina-Budish-O'Neill 2021 ressalva: depende do regime).
- 70% maker @ 2 bps + 30% taker @ 5 bps = **2.9 bps por side** vs 4.5 bps puro taker — economia ~3 bps/round-trip → +Sharpe ~0.08–0.12.
- Risco: 30% dos sinais "fortes" são exatamente quando a vela move e o post-only não enche → adverse selection no fallback. Mitigação: timeout curto (15–30s) + cap em desvio do close de referência.

**Status atual:** modelo entra no close[t] (= preço marcado), o que matematicamente é o **mid teórico do bar seguinte**, não o preço alcançável. Isso já embute ~1× spread de viés otimista que ninguém deduziu.

### Gap 3 — Latência cron→fill não modelada (impacto Sharpe realista −0.05 a −0.15)
GH Actions schedule cron: documentado pelo próprio GitHub como "may be delayed during periods of high loads of GitHub Actions workflow runs" — drift médio observado por usuários: 30s–3min, p99 até 15min (vide github/docs issues #14530, comunidade). `predict_4h.yml:8` agenda em `5 0,4,8,12,16,20` → assume runner sobe em <60s. Realista: p50 ~90s, p95 ~5min, p99 indefinido.

**Custo de stale signal**: usando vol realizada 4h média 2024 BTC ≈ 50 bps:
- 1 min stale → ~3 bps de drift esperado em magnitude (sign neutro mas custo igual a half-spread em RMS).
- 5 min stale → ~7 bps RMS.
- 10 min (p99 GH) → ~10 bps + risco de signal cancelado (próximo bar invalidaria).

**Para um modelo trade-rate ~2/sem com Sharpe 1.3 e PnL/trade ~50 bps líquido, perder 5 bps por latência = −10% no PnL líquido por trade = −0.10 a −0.15 no Sharpe**.

Mitigação: migrar predict para runner self-hosted (Hetzner/Oracle ARM cx22 ~€4/mês) OU acoplar o trigger ao receber WebSocket close de Binance kline (latência <2s). Treino caro pode ficar em GH cron, predict no edge.

## Experimento concreto

### Componente A — Modelo de slippage por size, calibrado em book real

1. **Coleta book L2 Binance spot+perp BTCUSDT por 30 dias**, snapshots a cada 60s + capture dos top-20 levels.
   - Endpoint: `GET /api/v3/depth?symbol=BTCUSDT&limit=100` (spot) e `GET /fapi/v1/depth` (perp). WebSocket `<symbol>@depth20@100ms` é o caminho produção.
   - Persist em `data/book_snapshots.parquet` (open_time, bid1, ask1, bid_qty_1..20, ask_qty_1..20, mid).
2. **Calibração η** do square-root law: regredir slippage realizado `(fill_px − mid_t) / mid_t` contra `σ_bar · √(X/V)` para diferentes sizes simulados (walk-the-book contra cada snapshot, com sizes [$100, $1k, $10k, $100k, $1M]).
3. **Validação:** comparar slippage simulada (walk-the-book) vs custo modelo √-law em janelas vol-normal vs vol-stress (vol > p90). Aceitar modelo se MAPE < 25%.

### Componente B — Benchmark exchange (custo efetivo realista)

Para cada exchange [Binance spot, Binance perp, Bybit perp, OKX perp, Hyperliquid perp]:

1. Coletar 7d de book snapshots + trades.
2. Simular execução dos últimos 200 sinais históricos do modelo dual-horizon como (a) puro taker e (b) maker-first com fallback taker @ 30s timeout.
3. Computar custo efetivo realizado = `taker_fee + maker_fee + half_spread + walk_book_impact + adverse_selection_proxy`.
4. Ranquear por **net Sharpe pós-custo-efetivo** + factor in funding (perps): funding negativo recorrente em short = custo extra.

**Saída esperada (hipótese):**
- Binance perp BTCUSDT vence em custo nominal mas paga funding 0.01%/8h em bull → 0.03%/dia ≈ 11% a.a. extra (long-only model em bull = paga). Spot evita funding mas perde 3 bps no maker.
- Hyperliquid: livro 50–80% mais fino → walk-book impact maior em size $50k+. Vence apenas em sizes <$10k.

### Componente C — Latência cron→fill, modelagem e mitigação

1. **Medição:** logar 4 timestamps por execução (cron_target, runner_pickup, predict_done, hypothetical_fill_ts) por 30d.
2. **Atribuição de drift:** `(close_px_at_predict − close_px_at_kline_close) / kline_close` × 1e4 bps. Espera-se distribuição centrada em zero (símile a noise) com std crescente em vol-stress.
3. **Decisão:**
   - Se mediana drift > 3 bps OU p95 > 15 bps → migrar predict p/ runner self-hosted ou kline-websocket trigger.
   - Manter treino em GH (custoso, off-critical-path).
4. **Tail risk hedge:** se elapsed > 4min após close_ts → **suprimir sinal desse bar** (signal age guard). Adiciona missed-trade cost, mas elimina cauda de slippage extrema.

### Critério de aceite (Sharpe pós-execução)

Re-rodar `exp_backtest_1k.py` com:
- Custo dinâmico via `compute_realistic_cost(size_usd, side, book_snapshot, vol_state)`.
- Latência amostral de distribuição GH Actions empírica (`simulate_latency_impact`).
- Filtro signal-age 4min.

**Targets ajustados:**
- Sharpe pós-custo-realista ≥ 1.0 (vs 1.29 atual otimista).
- Edge vs B&H líquido > 0 em ≥ 2 regimes (bull + bear/chop).
- Se Sharpe cair abaixo de 0.7 → modelo não é robusto à execução, voltar p/ Fase 5 (features) ou aceitar maker-first + signal-age guard como custos obrigatórios.

## Refs

1. Almgren R., Chriss N. (2000). *Optimal execution of portfolio transactions*. Journal of Risk 3(2). — square-root impact + linear permanent + risk-aversion trade-off.
2. Almgren R., Thum C., Hauptmann E., Li H. (2005). *Direct estimation of equity market impact*. Risk 18(7). — η ≈ 0.142 σ p/ equities; replicação em crypto rende 0.5–1.0.
3. Tóth B., Lemperiere Y., Deremble C., de Lataillade J., Kockelkoren J., Bouchaud J-P. (2011). *Anomalous price impact and the critical nature of liquidity in financial markets*. Phys. Rev. X 1, 021006. — square-root law universal.
4. Donier J., Bouchaud J-P. (2015). *Why do markets crash? Bitcoin data offers unprecedented insights*. PLoS ONE 10(10). — calibração de η em BTC (~0.5).
5. Cartea Á., Jaimungal S., Penalva J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press — cap. 6–8 sobre execution + adverse selection.
6. Cont R., Kukanov A. (2017). *Optimal order placement in limit order markets*. Quantitative Finance 17(1). — maker fill probability vs queue position.
7. Aquilina M., Budish E., O'Neill P. (2022). *Quantifying the High-Frequency Trading "Arms Race"*. QJE 137(1). — latency arbitrage e custo de stale quotes.
8. Hasbrouck J. (2009). *Trading costs and returns for U.S. equities: Estimating effective costs from daily data*. Journal of Finance 64(3). — Roll/Gibbs estimator (extrapolável p/ crypto com daily OHLC).
9. Kaiko Research (2024). *Bitcoin liquidity & slippage report*. https://research.kaiko.com — slippage empírico por exchange/size em BTC.
10. Amberdata (2025). *Crypto market microstructure: depth & impact 2024–2025*. https://www.amberdata.io/research — book depth comparado Binance/Bybit/OKX.
11. Binance fee schedule (consultar live): https://www.binance.com/en/fee/schedule — VIP0 spot 0.1% / 0.045% com BNB; futures maker 0.02% taker 0.05%.
12. Bybit fee schedule: https://www.bybit.com/en/help-center/article/Trading-Fee-Structure — VIP0 perp maker 0.02% taker 0.055%.
13. OKX fee schedule: https://www.okx.com/fees — Lv1 perp maker 0.02% taker 0.05%.
14. Hyperliquid docs: https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees — base taker 0.045%, maker 0.015%, vault rebates.
15. GitHub Actions schedule precision discussion: https://github.com/orgs/community/discussions/27130 — cron drift documentado, p99 minutos.
16. CCXT execution benchmarks: https://github.com/ccxt/ccxt — referência p/ unified slippage/maker logic em produção.
17. Hummingbot strategy docs (pure_market_making, cross_exchange_market_making): https://hummingbot.org/strategies/ — implementações production-ready de maker-first + spread/skew.
