# ROADMAP v2 — btc-forecast

> Reconciliação dos 7 briefs (microestrutura, validação, risk, alt-data, execution, infra, red-team).
> Forge mode. Prioridade = (lift × confiança) / (esforço × risco). Gate Red-Team antes de qualquer paper.
> Data: 2026-05-27.

---

## Veredito de estado

Projeto **acha** que está em Sharpe 1.29 (commit `bff285c`, dual-horizon AND). **Está**, após desconto honesto, em Sharpe 0.4–0.8 — provavelmente 0.6 ± 0.2. Custo subestimado em ~2× (8 bps vs 12–22 bps realistas), Sharpe anualizado infla 1.3–2.0× por overlap não-IID, threshold/AND/filtro-bear todos escolhidos in-sample. Telhado teórico pós-correção + sizing dinâmico + flow features: Sharpe 1.0–1.4 OOS honesto. Não estamos prontos pra paper. Não estamos prontos pra discutir infra. Estamos no meio da Fase 6 do ROADMAP original, com critério de morte defasado e validação inadequada.

---

## Resolução de contradições

1. **Risk vs Red-Team — Sharpe alvo 1.6–1.9 vs base real 0.4–0.8.** Red-Team vence. Risk Manager calibrou em cima de número fantasma. Lift de sizing dinâmico (vol-targeting + Grossman-Zhou) é real mas em base 0.6 entrega 0.9–1.3, não 1.9. Risk **só executa depois** de baseline ser recalibrado com custo correto + uniqueness weighting + holdout split.

2. **Quant Microestrutura vs Red-Team — +0.15–0.40 Sharpe de OFI/taker.** Microestrutura ganha em conteúdo (taker_buy do campo 9 do kline é grátis, OFI é literatura sólida Cont-Kukanov) mas seu lift estimado assume baseline 1.29. Recalibrado em base 0.6: esperar +0.05–0.20. Ainda alto ROI porque custo de implementação é triv. **Executa depois do gate Red-Team — features novas viram mais espaço pra overfit se baseline está bugado.**

3. **Execution vs Risk — custo 3× maior já invalida lift de Risk?** Sim e não. Execution diz custo real é 12–22 bps (vs 8 hardcoded). Isso derruba Sharpe e **também** derruba o benefício relativo de Kelly/vol-targeting porque trades pequenos viram negativos pós-custo. **Custo correto é PRÉ-REQUISITO de Risk.** Sem ele, sizing é cosmética.

4. **Alt-Data vs Quant Microestrutura — quem vem primeiro?** Quant ganha. Razões: (a) taker_buy_volume já vem no response do kline — zero novo endpoint, zero novo workflow. Alt-Data #2 (Binance OI) é exatamente o mesmo item que Microestrutura #2, mas Alt-Data quer começar pelo OI e Quant pelo taker_buy. Taker_buy custa 30 min de reparse; OI exige novo workflow GH Actions + backfill 30d rolling. **Ordem: taker_buy (Quant #1) → basis/OI (Quant #2 = Alt-Data #2) → DVOL/Coinalyze adiados.**

5. **Infra vs Red-Team — paper trading agora ou nunca?** Red-Team vence frontalmente. Infra propõe stack pré-paper (Hetzner, FastAPI, Prometheus, kill-switch) num projeto cujo Sharpe real provavelmente não bate 0.7. Construir paper trading em cima de sinal não-validado é queimar 5 dias de engenharia em algo que será derrubado. **Infra fica em parking lot até Sharpe pós-Red-Team > 0.7 com holdout limpo.** Único item de Infra que sobe pra Esse Mês: drift watchdog standalone (PSI/KS), porque ele também serve pra detectar leak/bug no próprio backtest.

6. **Validação vs todos — DSR/PSR/CPCV antes ou depois das features?** Validação vence parcial. Sem DSR + holdout split, **toda feature nova é roleta russa de overfit.** Mas CPCV completo + Romano-Wolf em todos exp_* é ~7 dias de trabalho. Compromisso: faz o **mínimo viável de validação** (bootstrap IC + PSR + holdout 2025+) ESSA SEMANA, e CPCV/Romano-Wolf vira marco de Esse Mês.

7. **Risk f_kelly vs Red-Team uniqueness.** Risk usa `p` do modelo direto pra half-Kelly. Red-Team mostra que `p` está sistematicamente otimista porque uniqueness não ponderada. **Aplicar weights primeiro, depois recalibrar `p` pra sizing.** Senão Kelly fica baseado em proba inflada → over-sizing.

8. **Execution latência GH Actions vs Infra VPS.** Mesmo problema, duas soluções. Execution mede; Infra migra. Ordem: medir primeiro (3 dias log de drift), só migra se p95 > 5 min ou mediana > 3 bps. Provavelmente migra, mas com dado em mão.

9. **Microestrutura Sharpe +0.15 vs Validação K_efetivo já em 50.** Cada feature nova multiplica K. Solução: pre-registration. **Antes de rodar OFI, escrever `research_log.md` com hipótese explícita + critério de sucesso fixo (Sharpe pós-uniqueness ≥ baseline + 0.10), e contabilizar +1 no K.**

10. **Red-Team MED-4 entry timing vs Risk equity-at-open.** Bug MED-4 (features t-1, entrada close[t]) tem que ser corrigido **antes** de Risk medir lift, senão Risk lift é medido contra baseline já contaminado.

---

## Esta semana (3 ações)

### A1. Recalibrar baseline com custo + holdout split + uniqueness weighting
- **Origem:** Red-Team HIGH-1, HIGH-3, HIGH-6, HIGH-9.
- **Arquivo/comando:**
  - Editar `notebooks/exp_backtest_1k.py:40`, `notebooks/exp_ensemble.py:38`, `notebooks/exp_multi_horizon.py:45`, `notebooks/exp_threshold_grid.py:47`, `notebooks/06_model_v2.py:38`, `notebooks/08_model_v3_sentiment.py:77`: trocar `COST = 0.0008` → `COST = 0.0015` (taker padrão sem BNB) e adicionar variante `COST_STRESS = 0.0022`.
  - Implementar `avg_uniqueness` em `pipeline/labels.py` (LdP eq.4.2: para cada label i com label_endtime, contar quantos outros labels overlapam). Passar como `weight=` em `lgb.Dataset` em `pipeline/model.py:_train_horizon` e em todos exp_*.
  - Dividir histórico: VAL = 2023-01 → 2024-12, HOLDOUT = 2025-01 → 2026-05. Refazer escolha de threshold + ensemble rule + NO_BEAR no VAL **apenas**, congelar, reportar HOLDOUT sem retoque.
- **Critério de sucesso:**
  - Sharpe HOLDOUT (bar-based, não trade-based) ≥ 0.7 líquido com COST=0.0015.
  - Profit factor HOLDOUT ≥ 1.2.
  - Bate B&H líquido no HOLDOUT (HOLDOUT B&H Sharpe é ~0.4–0.6 em 2025).
  - **Se Sharpe HOLDOUT < 0.5 → projeto entra em fase de morte controlada (ver §Critério de morte revisado).**
- **Tempo:** 3 dias úteis.

### A2. Adicionar taker_buy_ratio + OFI proxy (feature de fluxo zero-custo)
- **Origem:** Quant Microestrutura #1, Alt-Data #2 (parcial).
- **Pré-req:** A1 concluído (baseline honesto).
- **Arquivo/comando:**
  - `pipeline/binance.py:fetch_klines` — preservar `taker_buy_base_asset_volume` (campo 9) e `taker_buy_quote_asset_volume` (campo 10). Hoje são descartados. Reparse de `data/ohlcv_15m.parquet` precisa de refetch incremental dos últimos 30d (resto pode ser backfill de `data.binance.vision` dumps diários).
  - Novo `add_microstructure(df)` em `pipeline/features.py`: `taker_buy_ratio = taker_buy_base / volume`, `ofi_proxy = (2*taker_buy_base - volume) / volume`, Z-score rolling 7d/30d, `cvd = cumsum(2*taker_buy_base - volume)`. Plugar em `build_v2` antes de `apply_lag` (lag automático de 1).
  - Atualizar `research_log.md` (criar se não existir) com hipótese, K_atual, Sharpe alvo.
- **Critério de sucesso:**
  - Sharpe HOLDOUT (bar-based, COST=0.0015) ≥ A1 + 0.10.
  - SHAP importance de ≥ 2 features novas no top-20.
  - Profit factor HOLDOUT não cai > 5%.
  - Se Sharpe HOLDOUT < A1 + 0.05 → matar feature, registrar em research_log como K consumido sem ganho.
- **Tempo:** 2 dias úteis.

### A3. Implementar Probabilistic Sharpe Ratio + bootstrap IC e atualizar critério de morte
- **Origem:** Validação Estatística #1, #2.
- **Pré-req:** nenhum (paraleliza com A1, mas reporta DEPOIS de A1).
- **Arquivo/comando:**
  - Novo `pipeline/stats.py`: `psr(returns, sr_star=0)`, `psr(returns, sr_star=1.0)`, `bootstrap_sharpe_ci(returns, n_boot=5000, block_size=auto)` usando `arch.bootstrap.StationaryBootstrap`.
  - Plugar em `exp_backtest_1k.py` reportando `Sharpe = 0.X (CI95 [a, b]), PSR(0) = z, PSR(1) = w` em vez de só point estimate.
  - Atualizar §"Critério de morte" no ROADMAP: ver §Critério de morte revisado abaixo.
- **Critério de sucesso:**
  - Todo `exp_*.py` daqui pra frente reporta IC 95% e PSR(0).
  - Baseline pós-A1 tem PSR(0) > 0.95 OU projeto entra fase de morte.
- **Tempo:** 1 dia útil.

**Total semana: 6 dias úteis (cabe em uma semana corrida + 1d buffer).**

---

## Este mês (6 ações)

### M1. CPCV + Deflated Sharpe Ratio + PBO via CSCV
- **Origem:** Validação #3, #4, #5.
- **Pré-req:** A1, A3.
- **Arquivo/comando:** usar `mlfinpy.cross_validation.combinatorial.CombPurgedKFoldCV` (N=10, k=2 → 9 paths). Implementar DSR e PBO em `pipeline/stats.py`. Re-rodar baseline + A2 sob CPCV; reportar distribuição de Sharpe nos 9 paths.
- **Critério:** median Sharpe paths > 0.5, DSR > 0.6, PBO < 0.5. Se falha → mata feature ou volta pra baseline original.
- **Tempo:** 3 dias.

### M2. Custo dinâmico size-aware + maker-first sandbox + signal-age guard
- **Origem:** Execution Gap 1, 2, 3.
- **Pré-req:** A1 (baseline honesto).
- **Arquivo/comando:**
  - `pipeline/cost.py`: `realistic_cost(size_usd, side, regime_vol, book_snapshot=None)` — half_spread + η·σ·√(participation). Calibrar η com 7d de book L1 snapshots (`/api/v3/depth?symbol=BTCUSDT&limit=20`, cron 60s).
  - `predict_now.py` — adicionar `if (now - bar_close_ts) > 240s: skip_signal()` (signal-age guard, sugestão Execution).
  - Maker-first fica como **pesquisa offline** (simulação contra book snapshots), não vai pra prod ainda. Decisão de migrar pra perp Binance Futures (maker 2bps vs spot 4.5bps) fica registrada como ítem Q3.
- **Critério:** Sharpe HOLDOUT com custo dinâmico ≥ Sharpe HOLDOUT com COST=0.0015 fixo. Se cair > 0.1 → modelo é frágil a custo, voltar a otimizar features.
- **Tempo:** 4 dias.

### M3. Basis perp-spot + OI Δ + long/short ratio
- **Origem:** Quant Microestrutura #2, Alt-Data #2.
- **Pré-req:** A2 (taker_buy validado), M1 (CPCV antes de adicionar mais features).
- **Arquivo/comando:**
  - Novo workflow `.github/workflows/ingest_derivs_15m.yml` cron `*/15`: puxa `premiumIndex`, `openInterestHist?period=15m`, `topLongShortPositionRatio?period=15m` → `data/basis.parquet`, `data/oi_15m.parquet`, `data/lsr.parquet`.
  - `add_basis_oi(df)` em features.py: `basis = perp_close/spot_close - 1`, `oi_z30`, `oi_chg_4h`, `oi_price_divergence`, `lsr`, `lsr_z14`, interação `ix_oi_funding`.
  - Backfill: 30d rolling (Binance só serve isso) + cron contínuo a partir daqui. Walk-forward só nos folds com OI presente.
- **Critério:** Sharpe HOLDOUT (CPCV median) ≥ A2 + 0.10. SHAP top-20 inclui ≥ 1 feature dessa família.
- **Tempo:** 3 dias.

### M4. Sizing dinâmico — vol-targeting + drawdown gating (sem regime + sem Kelly por enquanto)
- **Origem:** Risk Manager #1, #3.
- **Pré-req:** A1, M2 (custo correto), M1 (uniqueness em `p` pra Kelly futuro).
- **Arquivo/comando:**
  - Schema bump em `pipeline/positions.py:open_position` — adicionar `size_pct`, `size_usd`, `equity_at_open`.
  - `pipeline/sizing.py`: implementar `f_vol` (target 25% vol anual) e `f_dd` (linear, DD_FLOOR -20%). **Pular `f_kelly` e `f_regime` neste sprint** — Kelly depende de `p` calibrado (M1 weights); regime depende de validação out-of-regime (não temos bear suficiente em HOLDOUT 2025).
  - Re-rodar `exp_backtest_1k.py` com sizing composto `size_pct = clamp(f_vol * f_dd, 0, 1)`.
- **Critério:** Calmar HOLDOUT ≥ baseline × 1.3, MaxDD HOLDOUT ≤ baseline × 0.7, Sharpe ≥ baseline. Se MaxDD não cai, vol-targeting está mal calibrado.
- **Tempo:** 3 dias.

### M5. Red-Team adversarial battery (shuffle / time-reversed / noise feature)
- **Origem:** Red-Team Apêndice + Testes adversariais 1-3.
- **Pré-req:** A1, A2 (testa o pipeline pós-correção).
- **Arquivo/comando:** rodar `proposals/red_team_tests.py` stub (já mencionado pelo brief; criar se não existir). Reportar Sharpe shuffle (esperado ≈ 0), Sharpe time-reversed (esperado < 0.3), noise feature importance (esperado < 0.01).
- **Critério:** todos 3 testes passam. **Se shuffle Sharpe > 0.3 → há leak não identificado, parar tudo e debugar.**
- **Tempo:** 1 dia.

### M6. Drift watchdog standalone + research_log.md formalizado
- **Origem:** Infra Gap #1, Validação passo 0 + 9.
- **Pré-req:** A1.
- **Arquivo/comando:**
  - `scripts/drift_watchdog.py`: PSI por feature (treino vs últimos 30d), KS-test, ADWIN opcional. Roda 1×/dia via GH Actions, posta no Telegram se PSI > 0.25.
  - `research_log.md`: lista cronológica de cada experimento com hipótese pré-registrada, K_efetivo cumulativo, Sharpe alvo, Sharpe observado, decisão (kill/keep). Política: **nenhum novo `exp_*.py` roda sem entrada no log antes.**
- **Critério:** watchdog roda 7 dias sem manual touch; research_log tem entrada retroativa pros 10+ experimentos já feitos com K honesto computado.
- **Tempo:** 2 dias.

**Total mês: 16 dias úteis (cabe em 4 semanas com folga).**

---

## Q3+ (parking lot)

- **Paper trading em testnet + executor + reconciliação Binance** (Infra Dia 3-5). Adiar até Sharpe HOLDOUT pós-CPCV > 0.8 com DSR > 0.7. Hoje seria construir prédio em areia movediça.
- **VPS Hetzner/Oracle + FastAPI + Prometheus + Grafana + Loki + kill-switch executável** (Infra Dias 1, 4, 5). Mesmo motivo. Drift watchdog standalone já cobre o risco P0 dessa categoria.
- **DVOL Deribit + put/call ratio** (Alt-Data #1). Lift estimado +0.05–0.15. Esforço médio (novo host, novo schema). Espera M3 entregar primeiro — se OI/basis sozinho mover agulha, DVOL fica pra Q3.
- **Coinalyze liquidations + LSR cross-venue** (Alt-Data #3). Mesma razão. Endpoint requer API key, ROI marginal vs já-coberto-em-M3.
- **On-chain (mempool fee, exchange netflow, miner outflow)** (Alt-Data gap #1). Granularidade ≥ 1h, ROI incerto em janela 4h. Q3+.
- **ETF spot flows Farside/SoSoValue** (Alt-Data gap #4). Daily, lag D+1. ROI baixo no grid 4h. Q3+.
- **VPIN bucketizado + CVD divergence** (Quant Microestrutura #3). Lift +0.05–0.15. Espera taker_buy + OI estarem dentro; VPIN exige tick-data ou bucketing custom, custo médio. Q3.
- **f_regime e f_kelly no sizing** (Risk #2, parte do #1). Espera M1 entregar `p` calibrado (uniqueness) e holdout bear period. Q3 quando 2026 bear (se vier) der amostra.
- **Maker-first em produção (Binance Futures perp)** (Execution Gap 2 prod). Espera paper trading. Q4.
- **Meta-labeling (`09_meta_labeling.py` com purge corrigido)** (Red-Team HIGH-7). Adiar — não está em produção, e baseline simples ainda precisa amadurecer.
- **Conformal prediction pro position sizing** (Validação passo 7). Q3 — depende de M1 + M4.
- **Hyperopt em produção (substituir hardcoded LGB_PARAMS pelos do `07_hyperopt.py`)** (Red-Team HIGH-8). Item barato, mas espera baseline pós-A1 estabilizar; senão otimiza em cima de bug.
- **SUS-1..SUS-5 (sentiment cross-source, hour/dow, yfinance retro, news_count breaks)**. Auditorias específicas, cada uma ~half day. Quando aparecer tempo morto.

---

## Combos achados

1. **Taker_buy_ratio + uniqueness weighting (A2 + A1 parte)** — sinergia oculta. Taker_buy é feature high-frequency com pouco overlap informacional consigo mesma; weighting por uniqueness vai derrubar peso dos labels antigos (técnicos) e amplificar peso dos labels jovens (microstrutura). Lift combinado potencial > soma das partes. **Custo: ambos já estão em Esta Semana, então combo é grátis.**

2. **Custo dinâmico (M2) + signal-age guard (M2) + vol-targeting (M4)** — trio que reduz cauda. Custo size-aware encarece trades em vol alta; signal-age corta entrada quando latência GH come a vela; vol-targeting reduz size em vol alta. Os três atacam o mesmo regime de stress por ângulos diferentes. Esperar Calmar +50% combinado vs +20-30% individual.

3. **Basis + funding + OI (M3) — squeeze detector composto.** Funding sozinho perde timing (já está no modelo, não move agulha). Basis sozinho é informacional mas ruidoso. OI Δ sozinho é confirmação atrasada. Os três juntos via interação `ix_oi_funding * basis_z` formam squeeze-risk score que provavelmente é o feature de maior SHAP no modelo M3. Listar essa interação explicitamente em `add_interactions`.

4. **Drift watchdog (M6) + research_log (M6) + Red-Team battery (M5)** — combo de governança. Os três custam ~4 dias somados, mas formam um "imune system" que impede recaída pros bugs HIGH-1 a HIGH-9. Sem eles, mês 3 vai redescobrir as mesmas armadilhas.

5. **Pre-registration explícita do K_efetivo no research_log + DSR (A3, M1, M6)** — o multiplicador honesto do projeto. Hoje K está em ~50–200 mascarado. Documentar isso é o que faz DSR ser ≥0.6 honesto vs DSR mentiroso. **Combo barato (≤1 dia incremental) com efeito gigante em credibilidade do número final.**

6. **Holdout 2025+ congelado + Romano-Wolf nos exp_* (A1 + M1)** — em vez de cada exp_* concorrer pelo "melhor Sharpe no pool inteiro", concorre pelo "melhor Sharpe em VAL com Romano-Wolf controlando FWER 5% em HOLDOUT". Provavelmente sobrevivem 1–3 experimentos dos 10+ feitos. Sinaliza honestamente onde está o edge real.

---

## Tabela de priorização (top-15)

Score = (lift_sharpe × confiança) / (esforço_dias × risco). Risco em escala 1 (certeza de ganho) a 5 (chance alta de gain ilusório). Confiança 0–1.

| # | Ação | Célula | Lift Sharpe | Esforço (d) | Risco | Confiança | Pré-req | Score | Janela |
|---|------|--------|-------------|-------------|-------|-----------|---------|-------|--------|
| 1 | Custo 0.0008→0.0015 + holdout split + uniqueness weighting (A1) | Red-Team | −0.3 a −0.5 (corretivo, eleva honestidade) | 3 | 1 | 0.95 | — | 0.16 (corretivo) | Esta semana |
| 2 | Taker_buy + OFI proxy (A2) | Quant μ | +0.05 a +0.20 | 2 | 2 | 0.70 | A1 | 0.044 | Esta semana |
| 3 | PSR + bootstrap IC (A3) | Validação | +0 (governança) | 1 | 1 | 0.95 | — | 0.95 (gov) | Esta semana |
| 4 | CPCV + DSR + PBO (M1) | Validação | −0.1 a −0.3 (corretivo) + governança | 3 | 1 | 0.90 | A1, A3 | 0.30 (gov) | Mês |
| 5 | Custo dinâmico + signal-age (M2) | Execution | −0.05 + tail-risk reduction | 4 | 2 | 0.80 | A1 | 0.020 | Mês |
| 6 | Basis + OI + LSR (M3) | Quant μ + Alt-data | +0.05 a +0.15 | 3 | 3 | 0.65 | A2, M1 | 0.022 | Mês |
| 7 | Vol-targeting + DD gating (M4) | Risk | +0.10 a +0.20 (Sharpe) + Calmar +50% | 3 | 2 | 0.75 | A1, M1, M2 | 0.038 | Mês |
| 8 | Red-Team adversarial battery (M5) | Red-Team | +0 (descoberta de bug) | 1 | 1 | 0.90 | A1, A2 | 0.90 (gov) | Mês |
| 9 | Drift watchdog + research_log (M6) | Infra + Validação | +0 (governança crítica) | 2 | 1 | 0.95 | A1 | 0.475 (gov) | Mês |
| 10 | DVOL + put/call ratio (Deribit) | Alt-data | +0.05 a +0.15 | 3 | 3 | 0.55 | M3 entregar | 0.018 | Q3 |
| 11 | f_kelly + f_regime sizing | Risk | +0.05 a +0.15 | 2 | 4 | 0.50 | M1, M4, bear sample | 0.009 | Q3 |
| 12 | VPIN + CVD divergence | Quant μ | +0.05 a +0.15 | 4 | 3 | 0.50 | M3 | 0.008 | Q3 |
| 13 | Paper trading testnet + executor | Infra | +0 (validação live) | 5 | 2 | 0.80 | Sharpe holdout > 0.8 | 0.08 (cond.) | Q3+ |
| 14 | VPS + FastAPI + Prom/Grafana stack | Infra | +0 + ops sanity | 5 | 2 | 0.85 | Paper aprovado | 0.085 (cond.) | Q4 |
| 15 | Coinalyze cross-venue liq/LSR | Alt-data | +0.05 a +0.10 | 3 | 3 | 0.50 | M3 + DVOL nulo | 0.008 | Q3 |

**Observações da tabela:**
- Itens 1, 3, 4, 8, 9 têm "lift Sharpe" baixo ou nulo mas **score altíssimo** porque são corretivos/governança — não inventam alpha, **preservam** o que existe e impedem que se invente alpha fantasma.
- Itens 7 (sizing) e 6 (basis/OI) são os candidatos a alpha real, mas só funcionam em cima de baseline limpo.
- Item 13–14 (paper + infra) têm score condicional — pra serem pagáveis, Sharpe pós-correção precisa cruzar o gate.

---

## Critério de morte revisado

Versão antiga (ROADMAP.md §2): "Sharpe rolling 90d < 0.3 por 4 semanas consecutivas em paper trading."

**Problema:** assume que paper trading existe e que Sharpe medido é confiável. Red-Team mostra que neither.

**Versão revisada:**

1. **Gate 1 — pós-A1 (esta semana):** Se Sharpe HOLDOUT (2025+) bar-based líquido com COST=0.0015 e weights por uniqueness < 0.5 → **projeto entra em estado terminal**. Razões válidas pra continuar mesmo assim: descobrir bug óbvio (não confundir com "achar outro experimento que dê Sharpe maior"). Tempo máximo de extensão: 2 semanas.

2. **Gate 2 — pós-M1 (mês 1):** Se median Sharpe nos 9 paths CPCV < 0.5, OU DSR < 0.6, OU PBO > 0.5 → **projeto morre**. Sem extensão. Esse gate substitui o "Sharpe rolling 90d" porque mede a coisa certa (Sharpe deflated + distribuição vs single path).

3. **Gate 3 — pós-M3 + M4 (mês 1 fim):** Se Sharpe HOLDOUT pós-features novas e sizing dinâmico não bateu A1 + 0.15 → as features de fluxo + sizing não estão entregando edge incremental. **Projeto não morre ainda** mas entra em fase "manutenção" — sem novos experimentos por 4 semanas, só monitoria. Se em 6 semanas drift watchdog não capturou nada interessante, encerra.

4. **Gate paper (Q3+, condicional):** Quando finalmente entrar em paper, retomar critério original: "Sharpe rolling 90d < 0.3 por 4 semanas". MAS agora "rolling 90d" tem que ser computado com bootstrap IC e o critério é "limite inferior IC95 < 0.0" — não point estimate.

5. **Gate de honestidade contínuo:** Toda nova `exp_*.py` consome 1 unidade de K. K acumula em `research_log.md`. Antes de cada exp: computar Sharpe haircut esperado (Harvey-Liu) — se upside pós-haircut < 0.7 → **não roda**. Disciplina mata mais alpha falso que qualquer feature engineering.

---

## O que você faz amanhã de manhã

1. Abre `pipeline/labels.py`, escreve `avg_uniqueness()` (LdP eq.4.2). 2h.
2. Edita os 6 arquivos com `COST = 0.0008` → `0.0015`. Grep + replace. 15 min.
3. Adiciona arg `weight=` em todos `lgb.Dataset(...)` em exp_* e em `pipeline/model.py:_train_horizon`. 1h.
4. Define VAL/HOLDOUT split em `exp_backtest_1k.py` (constantes `VAL_END = "2024-12-31"`, `HOLDOUT_START = "2025-01-01"`). 30 min.
5. Roda baseline com config nova, anota número em `research_log.md` (cria o arquivo). Esse é seu novo zero. 30 min execução + 10 min escrita.

Se Sharpe HOLDOUT vier < 0.5, **pare e debate Gate 1 antes de qualquer outra ação**. Não pule pra A2.
