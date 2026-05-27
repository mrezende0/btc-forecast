# Microestrutura — diagnóstico quant para btc-forecast

Sharpe atual 1.29 (dual-horizon AND ensemble, commit `bff285c`). Lente atual: OHLCV resampleado 15m/4h + funding rate diário e Z90. Toda a alpha vem de momentum/MAs/RSI/ATR + funding nível. **Nenhuma feature de fluxo de ordens, agressão (taker), basis perp-spot, OI ou divergência preço-volume direcional.**

## Diagnóstico

O `pipeline/features.py` faz uma boa engenharia técnica + macro + sentiment, mas opera em um regime de informação pobre — só preço fechado e volume não-direcional. Em perpétuos BTC, a distribuição de retornos a 15m é dominada por (a) iniciador da agressão (taker buy vs sell), (b) basis perp-spot e funding, (c) cascatas de liquidação. Nenhum dos três está no vetor. Pior: o `binance.py` puxa `klines` **spot via `data-api`** mas descarta os campos 9 e 10 do response — taker buy base/quote asset volume — que estão de graça e dão proxy direto pra OFI agregado por barra (Cont-Kukanov, JFinEcon 2014). Funding entra com `as-of backward` mas **basis spot-vs-perp**, `openInterestHist`, `topLongShortPositionRatio` e `takerlongshortRatio` (todos endpoints públicos `/futures/data/*`) não são consumidos. Easley/López de Prado/O'Hara (2012) e crypto-extensions 2024 mostram VPIN como proxy de toxicidade que prevê jumps. O modelo hoje vê o **resultado** (preço, vol realizada) mas não o **gerador** (quem agrediu o book). Edge intraday em crypto vive aí.

## Top-3 Gaps

### 1. Taker Buy Ratio + OFI agregado por barra (gap mais barato e maior expected value)

- **Problema:** features.py linha 88-91 calcula `vol_z7d`/`vol_z30d` mas trata volume como escalar não-direcional. Binance retorna `taker_buy_base_asset_volume` (campo 9 do kline). Razão `taker_buy / volume` é proxy direto de OFI agregado a 15m — quem foi market-taker. Cont, Kukanov & Stoikov (2014) provam relação linear entre OFI e retorno em janelas curtas, slope inverso à profundidade. Anastasopoulos & Gradojevic (2025, EFMA) replicam para crypto em 84 ativos e mostram que **order flow tem poder preditivo OOS para retornos crypto via ML não-linear**.
- **Impacto esperado:** Sharpe +0.15 a +0.40. Em equity intraday Cont-Kukanov mostra R² 0.65 vs returns em janelas de segundos; degrada com agregação mas 15m em crypto ainda preserva sinal direcional. Em crypto especificamente, Easley et al. (2024) reporta predictive power persistente.
- **Evidência:** Cont/Kukanov/Stoikov (JFinEcon 2014, "The Price Impact of Order Book Events", SSRN 1712822); Anastasopoulos & Gradojevic (EFMA 2025, "Order Flow and Cryptocurrency Returns"); Easley et al. (2024) recap em VisualHFT.

### 2. Basis perp-spot + OI delta + long/short ratio (regime detector funding-aware)

- **Problema:** `add_funding` usa só nível, Z90 e EMA. Não usa **basis perp-spot instantâneo** (perp_close / spot_close - 1) nem **ΔOI / Δprice** (sinal de novas posições vs liquidação). XT Exchange (Apr 2026) e Zeeshan Ali (SSRN 5611392, "Anatomy of Oct 10-11 2025 Cascade") mostram que **OI crescente + funding extremo + basis esticado** → cascata em janela 24-72h. Funding sozinho perde o timing.
- **Impacto esperado:** Sharpe +0.10 a +0.25 via filtro de regime (evita short-squeeze do lado long; evita long em pico de leverage). Reduz tail risk especialmente — max DD provável -30 a -50%.
- **Evidência:** Endpoints Binance `GET /fapi/v1/premiumIndex` (mark/index/funding em tempo real), `GET /futures/data/openInterestHist`, `GET /futures/data/topLongShortPositionRatio` — todos públicos, rate limit 1000 req/5min. Paper SSRN 5611392 (Zeeshan Ali, 2025); MDPI "Two-Tiered Structure of Cryptocurrency Funding Rate Markets" (2025); Hugonnier/Jermann arXiv 2310.11771 ("Perpetual Futures Pricing", 2024).

### 3. VPIN bucketizado por volume + CVD divergence (toxicidade + confirmação)

- **Problema:** sem proxy de informed-trading. Easley-López de Prado-O'Hara (2012) mostra VPIN como **leading indicator de stress/jumps** — picos de VPIN precedem flash crashes. Bitcoin específico: Bitcoin wild moves paper (ResearchGate 396478814, 2025) liga VPIN diretamente a price jumps. CVD divergence (preço HH + CVD LH) é confirmation tool clássica.
- **Impacto esperado:** Sharpe +0.05 a +0.15. Ganho maior em DD/calmar — VPIN filtra entradas em regime de toxicidade alta onde retorno esperado vira loteria.
- **Evidência:** Easley, López de Prado, O'Hara (Review of Financial Studies 2012, "Flow Toxicity and Liquidity in a High-Frequency World"); Bitcoin wild moves paper (Mendoza et al., 2025); VisualHFT VPIN ref.

## Experimento concreto

**Hipótese (H1):** adicionar 6 features de fluxo (taker_buy_ratio, taker_buy_ratio_z, ofi_proxy, basis_perp_spot, oi_delta_z, long_short_ratio) ao ensemble LGB+XGB v2 (4h) aumenta Sharpe walk-forward ≥ +0.15 vs baseline 1.29, **mantendo** max DD ≤ baseline.

**Método:**
1. Refit do `pipeline/binance.py` para preservar campos 9-10 (`taker_buy_base_asset_volume`, `taker_buy_quote_asset_volume`) já presentes no kline response — **zero custo de API**.
2. Backfill `data/oi_hist.parquet` (`GET /futures/data/openInterestHist?period=15m`, 30 dias rolling, paginado), `data/long_short.parquet` (`topLongShortPositionRatio`), `data/basis.parquet` (`premiumIndex`).
3. Adicionar `add_microstructure(df)` em features.py: `taker_buy_ratio = taker_buy_base / volume`, `ofi_proxy = (2*taker_buy_base - volume) / volume`, Z-score rolling 7d/30d, `cvd = cumsum(2*taker_buy_base - volume)`, `cvd_div = ret_4h - cvd_4h_normalized`.
4. As-of backward join de OI/basis/long-short (mesmo padrão de funding).
5. Rodar `exp_ensemble.py` com novas features. Walk-forward 2023Q1→2025Q4, purge=12.
6. Ablation: rodar com -taker_features, -basis_features, -vpin_features pra isolar contribuição marginal.
7. Métrica primária: **Sharpe líquido walk-forward**. Secundárias: Max DD, Calmar, n_sig/quarter (não deve cair > 30%).

**Dados necessários:**
- Refetch klines (já disponível, apenas reparse): ~2GB.
- `openInterestHist`: 30d rolling × 4 anos = backfill via chunks de 30d (rate limit ok). ~50MB.
- `premiumIndex`: idem.
- `topLongShortPositionRatio`: idem.
- **Custo zero, tudo endpoint público sem auth.**

**Métrica de sucesso:** Sharpe ≥ 1.45 (Δ +0.15 vs 1.29); Max DD ≤ baseline; ≥ 4 features de microestrutura com `gain importance` > median do baseline. Critério de morte do experimento: Sharpe < 1.20 (regressão) ou DD piora > 20%.

**Esforço:** 3-4 dias.
- D1: refetch + parse + 3 novos endpoints + storage (binance.py + ingest_15m.py).
- D2: `add_microstructure` em features.py + `add_basis_oi` + as-of joins + testes de leakage (shift rigoroso).
- D3: rerun `exp_ensemble.py` + ablation + SHAP por feature group.
- D4: brief com resultados + ROADMAP update.

## Refs

1. Cont, Kukanov & Stoikov (2014) — "The Price Impact of Order Book Events", *Journal of Financial Econometrics* 12(1):47-88. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1712822
2. Easley, López de Prado & O'Hara (2012) — "Flow Toxicity and Liquidity in a High-Frequency World", *Review of Financial Studies*. PDF: https://www.quantresearch.org/VPIN.pdf
3. Stoikov, S. (2018) — "The micro-price: a high-frequency estimator of future prices", *Quantitative Finance* 18(12):1959-1966. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694
4. Cartea, Jaimungal & Penalva (2015) — *Algorithmic and High-Frequency Trading*, Cambridge UP. https://assets.cambridge.org/97811070/91146/frontmatter/9781107091146_frontmatter.pdf
5. Anastasopoulos & Gradojevic (EFMA 2025) — "Order Flow and Cryptocurrency Returns". PDF: http://www.efmaefm.org/0EFMAMEETINGS/EFMA%20ANNUAL%20MEETINGS/2025-Greece/papers/OrderFlowpaper.pdf
6. Mendoza et al. (2025) — "Bitcoin wild moves: evidence from order flow toxicity and price jumps". https://www.researchgate.net/publication/396478814
7. Hugonnier, Jermann & Malamud (2024) — "Perpetual Futures Pricing", arXiv:2310.11771. https://arxiv.org/html/2310.11771v2
8. He, Manela, Ross & von Wachter (2022) — "Fundamentals of Perpetual Futures", arXiv:2212.06888. https://arxiv.org/abs/2212.06888
9. Ali, Z. (2025) — "Anatomy of the Oct 10-11, 2025 Crypto Liquidation Cascade", SSRN 5611392. https://papers.ssrn.com/sol3/Delivery.cfm/5611392.pdf?abstractid=5611392
10. MDPI Mathematics (2025) — "The Two-Tiered Structure of Cryptocurrency Funding Rate Markets". https://www.mdpi.com/2227-7390/14/2/346
11. Hybrid VAR-NN for OFI prediction (2024) — arXiv:2411.08382. https://arxiv.org/pdf/2411.08382
12. Gorsh, D. (2024) — "Optimizing Cryptocurrency Trading with Machine Learning: Predictive Analytics with Limit Order Book and Sentiment Data", SSRN 4867340. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4867340
13. Binance Futures API — Taker Buy/Sell Volume endpoint. https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Taker-BuySell-Volume
14. Binance Futures API — Open Interest Statistics + Long/Short Position Ratio. https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics + https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Top-Trader-Long-Short-Ratio
15. Hummingbot Issue #5409 — micro-price / order book imbalance research thread. https://github.com/hummingbot/hummingbot/issues/5409
16. crypto-lake/analysis-sharing — quantitative analyses on crypto orderbook data (extreme imbalance + short-term returns autocorr). https://github.com/crypto-lake/analysis-sharing
