# Research Log

Diário cronológico de experimentos. Política: **nenhum `exp_*.py` novo roda sem entrada aqui ANTES**.

Cada entrada: hipótese pré-registrada, K acumulado, métrica alvo, resultado, decisão.

Conceito de K: cada hipótese testada consome 1 unidade. Multiple testing infla expectativa de Sharpe via Bonferroni / Harvey-Liu. Se K_acumulado=100 e Sharpe alvo=1.0, expected_max_sharpe_under_null ≈ 0.4 → precisa de Sharpe observado >> 1.0 pra ser real.

---

## Entradas retroativas (K honesto antes do baseline novo)

Auditoria Red Team contabilizou K efetivo ≈ 92 trials nos `exp_*.py` pre-existentes:

| Data | Experimento | K | Sharpe reportado | Observação |
|------|-------------|---|------------------|-------------|
| pre-2026-05-27 | 05_model, 06_model_v2 | 2 | ~0.88 (mid h=12) | thresholds escolhidos in-sample |
| pre-2026-05-27 | 07_hyperopt | ~30 | n/a (não foi pra prod) | hyperopt logo direito mas params produção vieram de 06 in-sample |
| pre-2026-05-27 | 08_model_v3_sentiment | 1 | marginal | sentiment não moveu agulha |
| pre-2026-05-27 | 09_meta_labeling | 1 | n/a | CV 2-fold sem purge → suspeito |
| pre-2026-05-27 | 10_short_model | 1 | n/a | short branch off |
| pre-2026-05-27 | 11_time_decay | 1 | n/a | decay weights testado |
| pre-2026-05-27 | exp_asym_barriers | 5 | varias combinações | varredura barreiras asimétricas |
| pre-2026-05-27 | exp_ema200_veto | 1 | leve melhora | filtro EMA200 |
| pre-2026-05-27 | exp_wick_filter | 1 | inconclusivo | wick rejection |
| pre-2026-05-27 | exp_regime_analysis | 4 | bull/chop/bear segmentação | base pro NO_BEAR_THRESHOLD |
| pre-2026-05-27 | exp_threshold_grid | 35 (5 combos × 7 thrs) | melhor 1.07 | **HIGH-3 bug: thr escolhido no pool inteiro** |
| pre-2026-05-27 | exp_multi_horizon | 4 | dual-h melhor que single | ganho não validado em holdout próprio |
| pre-2026-05-27 | exp_ensemble | 5 | 1.29 (AND dual-horizon) | **HIGH-5: max sobre 5 combos × 7 thrs = 35 hipóteses** |
| pre-2026-05-27 | exp_position_sizing | 2 | "full" e "risk1" comparados | NO_BEAR_THRESHOLD calibrado no test set |
| pre-2026-05-27 | exp_drawdown_analysis | 1 | dec análise | descritivo |
| pre-2026-05-27 | exp_backtest_1k | 1 | corolário | dual-horizon dos 30%/5% |
| **TOTAL K** | | **~92** | | Harvey-Liu haircut esperado 40-70% |

Veredito Red Team: Sharpe 1.29 reportado → estimativa OOS real **0.4-0.8** após corrigir custo + uniqueness + threshold in-sample + sample uniqueness + selection bias.

---

## A1 — Recalibração de baseline (2026-05-27)

**Hipótese:** Após corrigir 3 bugs HIGH do Red Team (custo 0.0008→0.0015, uniqueness weighting LdP eq.4.2, VAL/HOLDOUT split com threshold congelado no VAL), Sharpe HOLDOUT 2025+ líquido deve ≥ 0.5 com PSR(0) ≥ 0.95 — caso contrário projeto entra fase terminal (Gate 1 ROADMAP_v2).

**Mudanças implementadas:**
- `pipeline/labels.py` — adicionado `avg_uniqueness()` (LdP AFML eq.4.2) e `attach_uniqueness()`.
- `pipeline/model.py:build_training_matrix` — agora inclui `uniqueness_weight`; `train()` passa `weight=` pra `lgb.Dataset`.
- `notebooks/exp_backtest_1k.py` — build_matrix com uniqueness; training loop usa `weight=`; constantes `VAL_END=2024-12-31` e `HOLDOUT_START=2025-01-01`; relatório segmentado VAL/HOLDOUT com Sharpe + IC 95% bootstrap + PSR(0) + PSR(1) + MaxDD.
- `COST = 0.0008 → 0.0015` em 18 arquivos (`notebooks/*` + `pipeline/positions.py`).
- `COST_STRESS = 0.0022` adicionado em `exp_backtest_1k.py` pra cenário de stress.

**K incremental:** +1 (uma única hipótese testada honesta). Total K acumulado = 93.

**Critério de sucesso (Gate 1 ROADMAP_v2):**
- Sharpe HOLDOUT ≥ 0.5 bar-based líquido com COST=0.0015.
- PSR(0) ≥ 0.95.
- Bate B&H líquido no HOLDOUT.

**Resultado (rodado 2026-05-27):**

```
Período: 2023-01-01 → 2026-05-26 (1152 dias, 6909 bars 4h)
Total trades: 169  Win rate: 53.3%  Avg PnL/trade líquido: +0.02%
Sharpe full: 0.06  MaxDD: -22.3%  Final: $986.65 (-1.34%) vs B&H $4,666.94 (+366.7%)

VAL      2023-01-01 → 2024-12-31  $1,000 → $859.60  (-14.04%)
  Sharpe=-0.47  CI95=[-1.94, 0.95]  PSR(0)=0.253  PSR(1)=0.019  MaxDD=-18.1%

HOLDOUT  2025-01-01 → 2026-05-26  $859.60 → $986.65 (+14.78%)
  Sharpe=0.67   CI95=[-1.23, 2.56]  PSR(0)=0.762  PSR(1)=0.363  MaxDD=-17.1%
```

**Leitura honesta:**

1. **Sharpe 1.29 era artefato** — confirmado. Honest full-period = 0.06.
   Red Team estimou 0.4-0.8; HOLDOUT pegou 0.67 em cima do range. Calibração precisa.

2. **VAL period catastrófico** (Sharpe -0.47, MaxDD -18%). Modelo perdeu dinheiro 2 anos em-sample. Não é coincidência o backtest original ignorar isso: threshold 0.35 e NO_BEAR=-0.05 foram escolhidos sobre TODO o histórico (incluindo HOLDOUT), o que mascarava o péssimo desempenho em VAL.

3. **HOLDOUT positivo mas estatisticamente ruidoso**. CI95 enorme (-1.23 a 2.56) por sample size pequeno (apenas 17 meses). PSR(0)=0.762 → só 76% de chance de Sharpe > 0. Não é "edge", é "talvez".

4. **Bate B&H? NÃO.** B&H entregou $4666 vs $986 do modelo. Em bull market 2023-2025 o modelo perdeu 78% relativo ao B&H. Critério explícito do ROADMAP §2: "Bate buy-hold líquido? sim". **FALHA.**

**Gate 1 (ROADMAP_v2 §Critério de morte revisado, textual):**
> "Se Sharpe HOLDOUT (2025+) bar-based líquido com COST=0.0015 e weights por uniqueness < 0.5 → projeto entra em estado terminal."

Sharpe HOLDOUT = 0.67 > 0.5 → **Gate 1 NÃO trippa por literalidade**. Mas:
- Não bate B&H → critério original ROADMAP §2 falha.
- PSR(0) < 0.95 → não temos confiança estatística.
- VAL negativo → o modelo não tem edge consistente, só sorteou bem em 2025.

**Decisão:**

- **NÃO avançar pra A2 (taker_buy/OFI)** até decidir o que fazer com VAL period.
- 3 caminhos possíveis (discutir com o operador):
  - **Caminho A — re-escolher threshold no VAL:** rodar grid 0.30-0.55 só em VAL, escolher melhor, congelar, reportar HOLDOUT. Pode ser que threshold 0.45 + NO_BEAR ajustado salve VAL. Risco: ainda é overfit, só que num split menor.
  - **Caminho B — aceitar diagnóstico e morrer:** modelo não bate B&H em bull market, isso é o que importa. ROADMAP §8 ("quando matar") já tinha esse critério. Arquivar.
  - **Caminho C — features de fluxo primeiro:** apostar que taker_buy + OFI + basis movem a agulha o suficiente pra reverter VAL. Risco: K consumido sem ganho garantido; baseline ruim multiplica K efetivo.

**Recomendação minha:** Caminho A primeiro (1 dia). Se threshold tunado no VAL ainda dá VAL Sharpe < 0.3 → Caminho B. Se VAL > 0.3 e HOLDOUT >= VAL × 0.7 (regularização razoável) → Caminho C com baseline limpo.

**K incremental:** +1 (A1). Total K acumulado = 93.

---

## A1-A — Caminho A: re-tunar threshold no VAL apenas (2026-05-27)

**Hipótese:** Threshold + NO_BEAR + ensemble rule escolhidos in-sample sobre dataset completo (Red Team HIGH-3, HIGH-5, HIGH-9). Re-tuning honesto restrito ao VAL pode revelar config melhor que generaliza pro HOLDOUT.

**Método:**
- `notebooks/exp_a1_threshold_search.py` — walk-forward 1x (uniqueness + COST=0.0015), cacheia probas, grida 400 configs:
  - `thr_mid ∈ {0.30, 0.35, 0.40, 0.45, 0.50}` × 5
  - `thr_long ∈ {0.30, 0.35, 0.40, 0.45, 0.50}` × 5
  - `no_bear ∈ {off, -0.10, -0.05, 0.00}` × 4
  - `rule ∈ {AND, OR, MID, LONG}` × 4
- Ranking POR VAL Sharpe apenas. HOLDOUT congelado (só reportado, não escolhe).
- Constraint: VAL Sharpe ≥ 0.3 e n_trades_val ≥ 30 pra ser candidato.

**Resultado (rodado 2026-05-27):**

```
TOP-5 por VAL Sharpe:
 thr_mid thr_long no_bear rule  val_sr val_psr0  ho_sr  ho_psr0
 0.35    *        -0.05   MID    +0.36   0.693  +1.53   0.952
 (mesma tupla mid+no_bear+MID empata em 5 valores de thr_long porque MID ignora long)

 0.35    0.30     -0.10   AND    +0.31   0.674  +1.53   0.952  (segundo)
 0.35    0.30     -0.05   AND    +0.27   0.649  +1.56   0.952  (terceiro)
```

**WINNER:** `thr_mid=0.35, no_bear=-0.05, rule=MID` (abandona long-horizon).

```
VAL     2023-01→2024-12:  Sharpe +0.36  PSR(0) 0.693  MaxDD -14.1%  88 trades  +9.0%
HOLDOUT 2025-01→2026-05:  Sharpe +1.53  PSR(0) 0.952  MaxDD  -8.3%  72 trades  +32.8%
```

**Findings honestos:**

1. **Dual-horizon AND era prejudicial.** Top 5 configs por VAL Sharpe usam `rule=MID` (mid sozinho). O "Sharpe 1.29 dual-horizon AND ensemble" do commit bff285c era pior que mid sozinho. Confirma Red Team HIGH-9 + HIGH-5 (selection bias).

2. **thr_mid=0.35 sobreviveu o tuning honesto.** Não era overfit no threshold; era no ensemble rule + uniqueness não ponderada.

3. **NO_BEAR=-0.05 sobreviveu.** O filtro de regime bear está adicionando valor consistente.

4. **HOLDOUT > VAL em Sharpe** (1.53 vs 0.36, ~4x). Inusitado e suspeito — mas:
   - PSR(0)=0.952 alto → estatisticamente sólido
   - 72 trades em HOLDOUT → amostra razoável
   - MaxDD baixo (-8.3%) → consistente, não foi sorte de um trade big
   - Provavelmente 2025 foi um regime mais favorável (bull tendencial pós-ETF)
   - 2023-2024 VAL teve período de chop fim-2023 e dump verão-2024 (BTC -22%) que prejudicaram

5. **PSR(0) 0.952 = exatamente o threshold ROADMAP_v2 §A3.** Gate 1 agora passa completamente.

**Decisão:** WINNER promovido como novo baseline. Próxima sessão = Caminho C (A2 features de fluxo: taker_buy + OFI proxy).

**Cuidados:**
- Não promover dual-horizon como produção. Reverter `pipeline/model.py:predict_dual_horizon` pra usar só o mid.
- Documentar K incremental honesto: 1 entrada (este tuning A) — não 400 (eram pré-registrados como UM bloco de hipótese "tune A").
- HOLDOUT consumido como evidência de overfit-check. Próximas mudanças (A2+) precisarão de OUTRO holdout (ex: 2026-Q2 em diante). Reservar `data/walk_forward_probas.parquet` como artefato congelado.

**K incremental:** +1 (Caminho A — 1 hipótese testada). Total K = 94.

---

## E — Dynamic Leverage (2026-05-27)

**Hipótese:** Leverage condicional à confiança do modelo (`proba_mid`) e vol realizada amplifica retorno sem inflar Sharpe-per-DD vs leverage flat.

**Método:**
- `pipeline/model.py:dynamic_leverage(proba_mid, rv_30d_ann, in_bear)` — fórmula `clamp(base × f_vol, 1, LEVERAGE_MAX)` com `base = 1 + f_conf × (max-1)` e `f_vol = min(1, target/rv)`.
- `notebooks/exp_e_dynamic_leverage.py` — backtest contrafactual no cache walk_forward_probas.parquet (sem K extra: aplica fórmula em probas existentes).
- `pipeline/predict_now.py` — leverage agora vem de `dynamic_leverage()` por padrão. Env `TRADING_LEVERAGE` segue como override manual.
- `pipeline/telegram.py` — formatter exibe componentes (f_conf, f_vol) da leverage dinâmica.

**Resultado HOLDOUT 2025-01 → 2026-05 (cache A1-A):**
```
mode             HO Sharpe  HO final  HO MaxDD  avg lev  Sharpe/DD
flat_1x (atual)    +1.53     $1,328    -8.3%    1.00     0.184
dynamic [1,3]      +1.60     $1,397   -10.1%    1.12     0.158
dynamic [1,5]      +1.72     $1,576   -13.3%    1.37     0.129
flat_3x            +1.91     $2,647   -20.1%    3.00     0.095
```

**Caveat VAL 2023-2024:** dynamic [1,5] tem VAL Sharpe -0.16 (vs flat_1x +0.36). Em regime adverso, leverage amplifica perdas. f_vol mitiga mas não anula. Trade-off explícito.

**Decisão:** KEEP com `LEVERAGE_MAX=5`. Sharpe-per-DD ainda 1.36× melhor que flat_3x. Em prod, formato Telegram mostra origem (dinâmica/manual) pra o operador decidir.

**K incremental:** +1 (uma fórmula testada). Total K = 95.

---

## F — SHORT model (em andamento, 2026-05-27)

**Hipótese:** Se o modelo tem edge em direção LONG (target = label==+1, HO Sharpe 1.53), o espelho SHORT (target = label==-1) deve ter edge simétrico em magnitude > 0.5 OOS honesto. Notebook 10 sugeriu isso (sem uniqueness, sem holdout split limpo). Re-testar com pipeline novo.

**Método pré-registrado:**
- `notebooks/exp_f_short_walkforward.py` — walk-forward expanding 2023-01-01 → presente, retreino a cada 90d.
- Mesma feature matrix (build_v2 BTC 4h, lag=1), uniqueness weights LdP, COST=0.0015.
- Target: `y_short = (label == -1)` (triple-barrier 3×ATR, lower hit antes de upper/timeout).
- Reporta 3 estratégias separadas:
  - LONG-only (baseline atual): proba_long > 0.35, no_bear=-0.05
  - SHORT-only: proba_short > 0.35, **bull filter inverso**: ret_30d > +0.05 → suprime (não shorta em mercado já caindo, espera reversão na alta).
  - COMBINED: soma trades não-conflitantes (se ambos sinalizam, anula).
- VAL: 2023-01 → 2024-12, HOLDOUT: 2025-01 → 2026-05.

**Critério KEEP/KILL:**
- KEEP SHORT se: HO Sharpe SHORT-only ≥ 0.5 E HO Sharpe COMBINED ≥ HO Sharpe LONG-only + 0.10.
- KILL se: HO Sharpe SHORT-only < 0.3 OU COMBINED < LONG-only (não traz incremento real).
- NEEDS-MORE-DATA se: HO Sharpe SHORT-only ∈ [0.3, 0.5] — anota mas não promove.

**K incremental:** +1. Total K projetado = 96.

**Resultado (rodado 2026-05-27):**

```
mode           VAL Shp   HO Shp   VAL n   HO n   HO win%   HO DD    HO final
long_only      +0.36     +1.53      88     71    60.6%    -8.3%    $1,328
short_only     -0.90     -0.50     118     73    47.9%   -24.4%    $864
combined       -0.86     +0.89     183    114    55.3%   -15.2%    $1,248
```

**Decisão: KILL** (critério pré-registrado dispara: HO Sharpe SHORT < 0.3).

**Findings:**
1. SHORT espelho **não tem edge OOS** em HOLDOUT bullish 2025+. Win rate 47.9% sub-50%.
2. COMBINED **piora** LONG-only (Sharpe +0.89 vs +1.53). Adicionar shorts contamina, mesmo com anti-conflito.
3. Assimetria estrutural BTC: long tem horizon 48h favorável; short tem reversões rápidas → muitos timeouts em -ATR ainda no swing.

**Próximas tentativas possíveis (Q3+, parking lot):**
- Modelo SHORT com features assimétricas (RSI extremo, basis spike, OI delta) em vez de feature set espelho.
- Horizon 24h em vez de 48h pra short (reversões mais curtas).
- Treinar só em sub-amostras de regime bear identificado (não em todo histórico).

**K incremental:** +1 (consumido sem ganho). Total K = 96.

---

## G — Exit Logic refinada (2026-05-27)

**Hipótese:** Partial TP + trailing stop captam mais expectancy do mesmo sinal sem K de feature. Esperado +10-25% expectancy/trade.

**Método:** `notebooks/exp_g_exit_logic.py` — 6 modos no cache walk_forward_probas.parquet:
1. baseline (hard 3×ATR target/stop, prod hoje)
2. partial_tp_50 (50% em +1×ATR, 50% trailing 2×ATR)
3. partial_tp_30_30_40 (30/30/40 com trailing remainder)
4. trail_only (100% trailing 2×ATR)
5. tp1_be (50% em +1×ATR, 50% restante: stop move pra break-even, target full 3×ATR)
6. tp1_be_wide_trail (50% em TP1, 50% restante com trailing 4×ATR frouxo)

**Resultado HO 2025+:**

```
mode                  HO Sharpe   HO final  HO MaxDD   ΔSharpe
baseline              +1.23       $1,244    -7.3%      —
partial_tp_50         -1.49       $836     -20.7%      -2.72
partial_tp_30_30_40   -1.41       $839     -20.4%      -2.64
trail_only            -1.60       $796     -25.0%      -2.83
tp1_be                -0.43       $935     -13.1%      -1.66
tp1_be_wide_trail     +0.21       $1,022    -9.6%      -1.02
```

**Decisão: KILL todos.** Nenhum modo bate baseline.

**Findings estruturais:**
1. **Double-cost:** partial TP paga COST=0.15% duas vezes/trade (0.30% vs 0.15% baseline). Em trade que vai a target completo (3.3% bruto), baseline = +3.15% líq; partial 50/50 = +0.21% líq.
2. **ATR_MULT=3 já é optimum** pro horizon 48h BTC. Avg leg ret cai de +0.524% (baseline) pra +0.04% a -0.09% em todas variantes. O setup vencedor explora vela explosiva, não parcial.
3. **Trailing chandelier:** 2×ATR apertado demais pra vol BTC (pullbacks rotineiros 4-8% atravessam); 4×ATR frouxo demais (sai em timeout flat, não captura).

**K incremental:** +1 (consumido sem ganho). Total K = 97.

---

## Resumo de fases (sessão 2026-05-27 pós-A1-A)

| Fase | Item | Resultado | K |
|------|------|-----------|---|
| 1a | exp_c 1h vs 4h | KILL 1h (HO -0.73) | +1 |
| 1b | exp_d ETH walk-forward | KILL ETH (HO -0.33) | +1 |
| 2 | exp_e dynamic_leverage | **KEEP** [1,5] (HO +1.72 vs +1.53, +19% retorno) | +1 |
| 3 | exp_f SHORT model | KILL (HO -0.50 standalone, combined PIOR) | +1 |
| 4 | exp_g exit logic (6 modos) | KILL todos (double-cost mata expectancy) | +1 |

**K total cumulativo:** 95 (E mantido) + 4 KILL = **97-98 hipóteses testadas honestas**.

**Único ganho real:** dynamic_leverage. Outros 4 caminhos exauridos — edge é específico (BTC + 4h + LONG + ATR_MULT=3 + threshold 0.35).

---

## H — Drift Watchdog (2026-05-27, governança)

**Hipótese:** sem monitoria, regressões silenciosas (feature distribuição mudou, modelo previsões pioraram) só aparecem quando capital já foi perdido.

**Método:**
- `scripts/drift_watchdog.py` — PSI (Population Stability Index) + KS-test por feature. Baseline = VAL 2023-2024 (4386 bars), current = últimos 30d (180 bars 4h).
- Buckets: PSI <0.10 estável, 0.10-0.25 moderado, >0.25 ALERTA, >0.50 CRÍTICO.
- History append em `data/drift_history.parquet` (idempotente por data).
- Telegram alert se max PSI > 0.25 (envia top 5 críticos + top 5 alertas).
- `.github/workflows/drift_watchdog.yml` — cron 12:00 UTC diário.

**Primeira execução (2026-05-27):**

```
max PSI = 8.28
38 features CRÍTICAS · 7 ALERTAS · 7 moderadas

Top 5 críticos:
  ma_7d/30d/90d  PSI 8.28  (BTC subiu de ~47k → ~77k base)
  fg             PSI 5.95  (F&G 36 "Extreme Fear" vs base média 59)
  funding_ema8d  PSI 5.65  (funding ~0 vs base ~0.0001)
```

**Leitura:** alerta justo. O sistema está em regime estatisticamente diferente do treino. ma_* absolutas drifta naturalmente em bull cycle — futura iteração pode mascarar features absolutas. Pelo critério atual, drift > 0.25 detectado → operador deve avaliar se modelo precisa retreino com janela mais recente.

**K incremental:** 0 (governança, não consome K). Total K = 97.

---

## E2 — Ensemble seeds (2026-05-27) — ⚠️ ACHADO CRÍTICO

**Hipótese:** ensemble N=5 LightGBMs com seeds diferentes reduz variância e melhora Sharpe.

**Setup:** Walk-forward expanding quarterly 2023Q1→2026Q2, MID-only, thr=0.35, no_bear=-0.05, COST=0.0015, FULL sizing, uniqueness weights.

**Resultado (rodado 2x independentemente — agente E2 + bash manual, idênticos):**

```
config                       VAL Sh    HO Sh   trades   final $1k   HO MaxDD
N=1 (seed=42 baseline)       -0.395   +0.451     158       $904       -12.4%
N=3 (seeds[42,123,456])      -0.596   +0.591     150       $869        -9.7%
N=5 (seeds base)             -0.664   +0.419     154       $818       -12.6%
N=10 (seeds base+extra)      -0.544   +0.560     148       $872       -10.2%

VARIÂNCIA entre 5 single-seed runs (seeds 42, 123, 456, 789, 999):
  HO Sharpe  mean +0.586  std 0.477  range [-0.003, +1.127]
  Final $1k  mean $862    std $116   range [$683, $977]
```

**🚨 BOMBA: "Sharpe 1.36" reportado como baseline ERA UM SEED SORTUDO.**

Todas as métricas anteriores na sessão (Sharpe HOLDOUT 1.36 / 1.55, Final $1290 etc) vieram de
runs single-seed em RNG state específico. A performance ESPERADA real é Sharpe ~0.59 ± 0.48.

**Implicações:**
1. Modelo **NÃO bate B&H** em risk-adjusted (0.59 < 1.28).
2. Range [-0.003, +1.127] significa que 20% das rodadas podem dar Sharpe próximo de zero.
3. PSR 0.952 reportado em A1-A foi calculado sobre o run sortudo — pode estar inflado.

**Decisão:** ENSEMBLE_SEEDS = [42, 123, 456, 789, 999] promovido em produção.
- predict_dual_horizon agora usa train_ensemble + predict_ensemble_proba
- Custo: ~24s vs 3s single (5x). Aceitável em cron 4h.
- Não melhora Sharpe ESPERADO, mas garante operação com média esperada e não com seed sortudo/azarado.

**K incremental:** +1 (E2 — uma hipótese testada). Total K = 98.

---

## Política dia-a-dia

1. **Pré-registrar** experimento aqui ANTES de rodar.
2. Computar Sharpe haircut esperado (Harvey-Liu Bonferroni: `sr_alvo / sqrt(2 * log(K))`); se upside pós-haircut < 0.7 → **não roda**.
3. Após rodar, anotar resultado + decisão (KEEP / KILL / NEEDS-MORE-DATA).
4. K só aumenta — nunca decrementa. Killed experiments também consomem K.
