# Infra & Live Ops — btc-forecast → paper trading testnet

> Transição de "GH Actions cron + Telegram + git push de parquet" para um loop autônomo, observável,
> com drift watchdog e kill-switch real, hospedado num VPS único.

## Diagnóstico

**Estado atual (visto no repo):**

- **Ingestão:** GH Actions cron `*/15 * * * *` (`ingest_15m.yml`) e `0 6 * * *` (`ingest_daily.yml`).
  Cada run dá `pip install`, baixa Binance/yfinance/F&G, escreve parquet, `git add data/ && git push`.
- **Inferência:** `predict_4h.yml` roda `5 0,4,8,12,16,20 * * *` → `pipeline.predict_now` retreina
  dual-horizon LightGBM **a cada inferência** (build matriz + train mid + train long), prediz a
  última vela, append em `data/signals.parquet`, opcionalmente abre posição em `positions.parquet`
  e envia Telegram.
- **Monitor de saída:** `monitor_positions.yml` cron `*/15` → varre `positions.parquet`, fecha em
  target/stop/timeout, manda Telegram.
- **Dashboard:** Cowork lê `dashboard_state.json` commitado pelo `ingest_daily`.
- **Persistência:** TUDO em parquet versionado no git. Race conditions resolvidas com `git pull --rebase --autostash`.
- **Sem:** drift detection, métricas, structured logging, health check, kill-switch automatizado,
  paper trading (não há execução nem reconciliação com testnet), model registry, alertas além de
  Telegram, state store transacional.

**Riscos imediatos para paper trading 3–6 meses:**

1. **GH Actions não é serviço.** SLA não-garantido, runs atrasam minutos, e cron é "best-effort"
   (GitHub explicitamente avisa). Para `*/15` o jitter já mordeu na fase de ingest; em loop
   paper-trade isso vira ordens disparadas no momento errado.
2. **Git como banco de dados.** `signals.parquet` + `positions.parquet` num repo público versionado
   é cómico mas frágil: rebase manual em conflito, push 401, race entre `monitor` e `predict_now`
   tocando o mesmo arquivo no mesmo minuto. Já dependemos de `--autostash` pra não quebrar.
3. **Retreina a cada inferência.** `predict_dual_horizon()` chama `build_v2_from_parquets()` duas
   vezes e treina dois LightGBM do zero (`N_ROUNDS=500`) a cada 4h. Custo de GH minutes cresce
   linear e — pior — não há separação entre artefato de modelo e código. Sem model registry, sem
   reproducibility de uma predição passada.
4. **Sem drift watchdog.** ROADMAP fase 8 cita "Drift detection (KS/PSI nas features)" como TODO.
   Sem isso, modelo silenciosamente vai pra distribuição nova (regime shift BTC, mudança de
   funding regime, evento macro) e a única notícia é Sharpe colapsando 4 semanas depois — que é
   o critério de morte. Drift watchdog deve disparar antes do PnL.
5. **Sem kill-switch real.** O critério de morte ("Sharpe rolling 90d < 0.3 por 4 semanas") é
   doc, não código. Nada no `predict_now.py` checa Sharpe rolling antes de gerar sinal. Em paper
   é só auto-engano; em live (mesmo testnet) é dinheiro queimando.
6. **Telegram é o único alerta.** Sem pager hierarchy, sem ACK, sem dedup. Se o bot cair você
   descobre porque "parou de chegar mensagem", e em sistema que só envia em sinal positivo isso
   é silent failure clássico.
7. **Sem observability.** Não há latência de predição, taxa de erro Binance, lag entre vela
   fechada e sinal gerado, tempo de download da OHLCV. Quando algo quebrar, o único debug é
   abrir o run log do GH Actions.
8. **Single asset / single chave.** OK por escopo, mas combinado com 1–6 deixa zero margem.
   Se Binance API der 451 (já aconteceu — ver commit `55dfe59`), o paper trade morre silencioso.

## Top-3 Gaps

### Gap #1 — Sem drift watchdog (P0)

O modelo treina em ~2021–presente e prediz na vela mais recente, sem nenhuma verificação se
features em produção continuam parecidas com as do treino. PSI e KS-test em features-chave
(`funding_z90`, `rv_1d`, `dist_ma_30d`, `dxy_z30`, `fg`, `vix_z30`) precisam rodar pelo menos
1×/dia, com gates: PSI < 0.1 OK, 0.1–0.25 warn (Telegram), > 0.25 trip the breaker — congela
sinal e força revisão. ADWIN online opcional pra detectar shift mid-stream.

### Gap #2 — Sem serviço persistente + state store (P0)

GH Actions é build runner, não orquestrador. Paper trade precisa de:
(a) loop sempre-on (predictor FastAPI / scheduler APScheduler dentro de um container);
(b) state store transacional (SQLite WAL ou Redis) pra `positions` e `signals`, com lock e
audit log; (c) reconciliação periódica com Binance testnet (posição aberta no broker bate com
o estado interno?).

### Gap #3 — Sem observability + kill-switch executável (P1, mas dispara junto com #2)

Métricas Prometheus (predict_latency, signal_count, open_positions, pnl_realized_rolling_90d,
psi_max, binance_api_errors_total), Loki pra logs JSON, Grafana dashboard, e um job
`kill_switch.py` que lê `pnl_realized_rolling_90d` + `psi_max` e seta uma flag (`STATE.killed=1`)
que o predictor checa antes de emitir qualquer sinal. Trip = manda alerta Telegram **e** PagerDuty/email.

## Arquitetura alvo

```
                         ┌─────────────────────────────────────────────────┐
                         │  VPS único (Hetzner CX22 4€/mês ou Oracle Free) │
                         │   docker-compose up -d                          │
                         └─────────────────────────────────────────────────┘

  Binance Spot/Fut ─┐                                                 ┌──> Telegram (alert P1)
  yfinance/F&G/GDELT ├──> [ingestor]──(parquet vols mount)            ├──> Email/PagerDuty (P0)
  CoinDesk          ─┘        │                                       │
                              ▼                                       │
                       ┌────────────┐    features         ┌──────────────────┐
                       │  feature   │────────────────────>│  drift_watchdog  │
                       │  builder   │  parquet/duckdb     │  PSI/KS (cron)   │
                       │ (cron 4h)  │                     └────────┬─────────┘
                       └─────┬──────┘                              │ psi_max metric
                             │                                     ▼
                             ▼                              ┌────────────┐
                       ┌────────────┐    POST /predict      │ prometheus │
                       │ predictor  │<─────────cron─────────│            │
                       │ FastAPI    │                       └─────┬──────┘
                       │ (LGBM    ) │      ┌──────────┐           │
                       │ + scheduler│<────>│  redis   │           │ scrape
                       │  4h        │      │ state    │           ▼
                       └─────┬──────┘      │ killflag │     ┌────────────┐
                             │             │ positions│     │  grafana   │
                             │             └──────────┘     │ dashboards │
                             ▼                              └────────────┘
                       ┌────────────┐                              ▲
                       │ executor   │──signed REST──> Binance      │ logs
                       │ (paper:    │   testnet (fapi-testnet)     │
                       │  testnet)  │<──reconcile────              │
                       └─────┬──────┘                              │
                             │                                     │
                             └────────── JSON logs ───────> Loki ──┘
                                              │
                                              ▼
                                      ┌────────────────┐
                                      │ kill_switch.py │  cron 1h
                                      │ checks:        │
                                      │  - PSI > 0.25  │──> SET redis killed=1
                                      │  - Sharpe90d   │      Telegram P0
                                      │    < 0.3 × 4w  │
                                      │  - api_errors  │
                                      │    > N/h       │
                                      └────────────────┘
```

**Fluxo de dados (resumo):**

1. `ingestor` (cron container) puxa OHLCV/funding/macro/F&G/news → `/data/*.parquet` (volume).
2. `feature_builder` materializa features no schedule do bar (4h) → grava em DuckDB
   (`/data/features.duckdb`) + emite event no Redis pub/sub.
3. `predictor` (FastAPI uvicorn + APScheduler) consome event, carrega modelo do registry
   (`/models/lgbm_dual_v{N}.txt` versionado), chama `predict_dual_horizon()`, escreve sinal no
   Redis + parquet, expõe `/metrics`.
4. `executor` (paper mode = testnet) lê sinais, valida pre-trade risk checks (kill flag,
   exposição máxima, distância stop sane), envia ordem na testnet Binance, persiste fill.
5. `drift_watchdog` (cron 1h) recalcula PSI/KS na janela 30d vs treino, exporta gauge.
6. `kill_switch.py` (cron 1h) cruza métricas → seta `state.killed` no Redis se trip.
7. Prometheus scrapeia tudo, Grafana mostra, Loki guarda logs JSON estruturados.

**Por que essa shape:** mínimo viável que destrava paper trading sem virar Kubernetes-grade.
Tudo num docker-compose; backup = `tar` dos volumes; observabilidade self-hosted barata
(Hetzner CX22 cobre RAM/CPU sobrando, Oracle Cloud Free 4-core ARM idem).

## Experimento concreto — 3 a 5 dias pra testnet rodando

**Dia 1 — Migrar inferência pra container local + state store**

- Copiar `pipeline/predict_now.py` pra serviço FastAPI (`predictor/app.py`) com endpoints
  `/predict` (POST, idempotente por `bar_open_time`), `/healthz`, `/metrics`.
- Substituir `data/positions.parquet` + `data/signals.parquet` por SQLite WAL local
  (`pipeline/storage_sqlite.py`) — schema igual, transações ACID. Manter export parquet
  como artefato analítico (write-through).
- `docker compose up predictor redis prometheus grafana` localmente. Bate `/predict` no horário
  da vela, valida que sinal sai idêntico ao atual. Telegram continua funcionando.
- Critério: 1×/4h por 24h sem manual touch.

**Dia 2 — Drift watchdog + métricas**

- Implementar `scripts/drift_watchdog.py` (stub no `proposals/`): carrega features atuais (30d)
  vs reference (90d–365d antes), PSI por feature, gauge no Prometheus.
- Definir reference janela: features do último train confiável (período onde Sharpe walk-forward
  > 0.7). Salvar como `data/reference_features.parquet`.
- Grafana dashboard: PSI por feature (heatmap), predict_latency p95, signal_count_7d.
- Critério: drift_watchdog rodando 1×/h, alerta Telegram simulado (force-trigger feature
  manualmente alterada).

**Dia 3 — Binance testnet + executor paper**

- Conta `testnet.binancefuture.com`, API key/secret em `.env`. Usar `ccxt` com
  `exchange.set_sandbox_mode(True)` ou direct `https://testnet.binancefuture.com`.
- `executor/paper.py`: consome sinal do Redis, valida pre-trade (kill flag OFF, max position 1,
  stop dentro de N×ATR), envia MARKET buy + STOP-LOSS reduce-only + TAKE-PROFIT reduce-only.
  Reconciliação: 1×/min compara `exchange.fetch_positions()` com SQLite.
- Critério: 1 trade end-to-end em testnet (fake money) com fill confirmado e reconciliação OK.

**Dia 4 — Kill-switch + alerting**

- `scripts/kill_switch.py`: lê SQLite (`positions`), calcula Sharpe rolling 90d
  (window=120 trades ou 90d, o que vier primeiro), lê `psi_max` do Prometheus via HTTP,
  conta `binance_api_errors_total` 1h. Se trip → `redis.set("killed", 1, ex=86400)` +
  Telegram P0 + (futuro) email via SMTP.
- Predictor checa `redis.get("killed")` antes de emitir sinal. Se 1, loga e retorna sem ação.
- Critério: forçar PSI > 0.25 manualmente, validar que sinal seguinte é bloqueado.

**Dia 5 — Deploy VPS + observability**

- Hetzner CX22 (4€/mês, 2 vCPU 4GB ARM) ou Oracle Free 4-core ARM. Apenas SSH key, UFW
  (22, restrito IP), Tailscale opcional.
- `git clone` repo, `cp .env.example .env`, `docker compose up -d`.
- Grafana exposto via Caddy + Basic Auth (não público sem auth, NUNCA).
- Backup nightly: `restic` ou `rsync` dos volumes `/data` e `/models` pra B2/R2 (5–10€/ano).
- Critério: predictor + monitor + drift_watchdog + kill_switch rodando 72h sem manual touch,
  PnL paper acumulado visível no Grafana.

**O que NÃO fazer ainda:** mover ingest_15m do GH Actions (deixa onde está mais 1 sprint),
não criar feature store dedicado (DuckDB embebido basta), não Kubernetes, não Airflow.

## Refs

1. Huyen, Chip — *Designing Machine Learning Systems*, cap. 8 "Data Distribution Shifts and Monitoring".
   https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html
2. Huyen, Chip — Resumos cap. 8 (community).
   https://github.com/serodriguez68/designing-ml-systems-summary/blob/main/08-data-distribution-shifts-and%20monitoring-in-production.md
3. Yan, Eugene — *More Design Patterns For Machine Learning Systems*.
   https://eugeneyan.com/writing/more-patterns/
4. Yan, Eugene — Tag "production" (índice).
   https://eugeneyan.com/tag/production/
5. Shankar, Shreya — *Towards Observability for Production Machine Learning Pipelines* (arXiv 2108.13557).
   https://arxiv.org/abs/2108.13557
6. Shankar, Shreya et al. — *Operationalizing Machine Learning: An Interview Study* (arXiv 2209.09125).
   https://arxiv.org/abs/2209.09125
7. Fiddler AI — *Measuring Data Drift with the Population Stability Index (PSI)* (thresholds 0.1 / 0.2 / 0.25).
   https://www.fiddler.ai/blog/measuring-data-drift-population-stability-index
8. arXiv 2404.18673 — *Open-Source Drift Detection Tools in Action: Insights from Two Use Cases*
   (Evidently vs Alibi-Detect vs NannyML benchmark).
   https://arxiv.org/abs/2404.18673
9. Comparativo Evidently / Alibi-Detect / NannyML / WhyLabs / Fiddler.
   https://medium.com/@tanish.kandivlikar1412/comprehensive-comparison-of-ml-model-monitoring-tools-evidently-ai-alibi-detect-nannyml-a016d7dd8219
10. River-ml docs — KSWIN (Kolmogorov-Smirnov windowing online drift).
    https://riverml.xyz/dev/api/drift/KSWIN/
11. Binance Futures Testnet — base URL e setup.
    https://testnet.binancefuture.com/
12. CCXT issue #22978 — testando bot na Binance Futures Testnet (gotchas reais).
    https://github.com/ccxt/ccxt/issues/22978
13. Hummingbot — Paper Trade mode (referência de design pra paper executor).
    https://hummingbot.org/client/global-configs/paper-trade/
14. Dragonfly blog — *Building a Feature Store with Feast, DuckDB, and Dragonfly* (DuckDB como
    offline store leve, alternativa a Feast cheio).
    https://www.dragonflydb.io/blog/building-a-feature-store-with-feast-duckdb-and-dragonfly
15. Feast docs — DuckDB offline store oficial.
    https://docs.feast.dev/reference/offline-stores/duckdb
16. trallnag — *prometheus-fastapi-instrumentator* (lib que vamos usar no predictor).
    https://github.com/trallnag/prometheus-fastapi-instrumentator
17. Grafana docs — Install Loki with Docker Compose.
    https://grafana.com/docs/loki/latest/setup/install/docker/
18. blueswen/fastapi-observability — referência de stack FastAPI + Prometheus + Loki + Tempo.
    https://github.com/blueswen/fastapi-observability
19. Fowler, Martin — *CircuitBreaker* (pattern original, base do kill-switch).
    https://martinfowler.com/bliki/CircuitBreaker.html
20. Rulematch — *Kill Switch* em trading (semântica: bloqueio de entrada + cancel all).
    https://www.rulematch.com/trading/kill-switch/
