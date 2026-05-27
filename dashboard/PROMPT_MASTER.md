# Cowork Dashboard — BTC Forecast

## Como usar

1. Abrir Cowork (claude.ai/download)
2. Modo Cowork + connector **CoinDesk** ativo
3. Nova conversa, cola o prompt abaixo, Enter
4. Artifact "BTC Briefing" aparece na barra lateral
5. Toda manhã: abre o artifact + Reload → dados frescos

O dashboard consome **dois inputs**:
- `dashboard_state.json` do GitHub (auto-atualizado pelo workflow `ingest_daily`)
- **CoinDesk MCP** ao vivo pra manchetes

---

## Prompt master (copia e cola)

```
Você vai construir um artifact HTML persistente chamado "BTC Briefing" — meu
dashboard pessoal de leitura matinal do mercado de Bitcoin. Linguagem:
português brasileiro coloquial. Tudo dentro do Cowork.

PASSO 0 — VERIFIQUE CONECTORES
Chame mcp__mcp-registry__list_connectors com keywords ["coindesk"].
Se CoinDesk não estiver conectado, pare e me oriente.

PASSO 1 — CARREGUE O STATE JSON
O backend deste dashboard é gerado pelo meu repo github.com/MRezende0/btc-forecast.
Faça fetch de:
  https://raw.githubusercontent.com/MRezende0/btc-forecast/main/data/dashboard_state.json
Parse o JSON. Estrutura esperada:
{
  generated_at, price {last_close, last_dt, ret_24h, ret_7d, ret_30d, ret_90d, high_30d, low_30d},
  vol {rv_1d_ann, rv_1w_ann, rv_30d_ann},
  funding {last, last_dt, mean_30d, z_30d, q05_30d, q95_30d},
  macro {last_date, dxy:{last,z_30d,chg_5d}, vix:{...}, spx:{...}},
  fg {last, last_class, last_date, chg_7d, chg_30d},
  sentiment_news {available, last_date, net_sentiment_today, net_sentiment_7d_avg, news_count_today, news_count_7d_avg},
  data_health {ohlcv_15m_rows, funding_rows, macro_days, fg_days, news_days}
}

Mostre claramente em "Diagnóstico" no rodapé:
- generated_at (idade do snapshot)
- data_health (volumes coletados)
- Se algum bloco vier available:false, mostre o motivo em vez de inventar

PASSO 2 — DESCUBRA O TOOL COINDESK
Use ToolSearch pra encontrar o tool de fetch_news do CoinDesk
(geralmente "fetch_news" no MCP do CoinDesk). Chame com {limit: 30, lang: "EN"}
pra testar shape do response. Espere algo como:
  {Data: [{TITLE, URL, PUBLISHED_ON (unix segundos), SOURCE_DATA:{NAME},
          SENTIMENT ("POSITIVE"|"NEUTRAL"|"NEGATIVE"), CATEGORY_DATA:[{CATEGORY}]}]}

PASSO 3 — CONSTRUA O HTML EM 8 SEÇÕES
HTML autocontido (CSS inline), light mode, tipografia
-apple-system/BlinkMacSystemFont/"Segoe UI"/Helvetica. Cores:
  positivo #1d9e75, negativo #c93a3a, neutro #666.
  Badges fundo claro: pos #eaf3de/#27500a, neg #fcebeb/#791f1f, neu #f0efea/#444.

Seções nesta ordem:

(1) HEADER — "BTC Briefing" + última atualização (generated_at humanizado:
    "atualizado há 3h"). Indicador grande do preço atual e variação 24h.

(2) PREÇO E ESTRUTURA
    - Preço atual ($75.4k formato BR)
    - Variações 24h/7d/30d/90d com pills coloridas
    - Range 30d: low — high com marcador da posição atual
    - Texto curto: "BTC está X% acima/abaixo do meio do range 30d"

(3) VOLATILIDADE — comparação 1d vs 1w vs 30d
    - Se rv_1d > rv_30d * 1.3, badge "vol em alta"
    - Se rv_1d < rv_30d * 0.7, badge "vol comprimindo"
    - Implicação operacional curta (1 frase)

(4) DERIVATIVOS (FUNDING)
    - Funding atual em bp
    - Z-score 30d com pill: |z|>2 = extremo, |z|>1 = moderado
    - Interpretação: "long demais" (z alto), "shorts pesando" (z baixo)
    - 1 frase de leitura tática

(5) MACRO (D-1)
    - DXY, VIX, SPX em tabela: nível, Z-score 30d, mudança 5d
    - Highlight automático quando VIX z>1 ou SPX z<-1 (risk-off)
    - 1 frase: "ambiente macro X com viés Y pra cripto"

(6) SENTIMENTO
    - Fear & Greed: número, classe, mudança 7d, 30d
    - Se sentiment_news.available: net_sentiment hoje + média 7d
    - Se não: mostre badge "sentiment news indisponível"

(7) MANCHETES (CoinDesk MCP — ao vivo)
    - Pegue 15 manchetes mais recentes via fetch_news
    - Filtre só últimas 48h
    - Lista com: pill de sentimento + título clicável + fonte + tempo relativo
    - Conta breakdown: X positivas, Y neutras, Z negativas

(8) O QUE OLHARIA HOJE (síntese)
    - Use askClaude com Haiku (timeout 15s, prompt em pt-BR) cruzando:
      preço, vol regime, funding z, F&G, macro, sentimento manchetes
    - 3-5 bullets diretos, sem hedging
    - FALLBACK determinístico (computado em JS puro): se F&G<30 OR funding_z>1.5
      OR vix_z>1, gera bullets fixos baseado nesses gatilhos.
    - NUNCA deixe vazio.

(9) DIAGNÓSTICO (rodapé colapsável)
    - generated_at, idade do snapshot
    - data_health (linhas/dias por dataset)
    - Tools chamados + qualquer erro

REGRAS IMPLEMENTAÇÃO
- TODAS chamadas MCP via window.cowork.callMcpTool(name, args).
- Unwrap robusto: formato MCP padrão é {content:[{type:"text",text:"..."}]} → JSON.parse.
- TODAS chamadas Haiku via window.cowork.askClaude(prompt, data[]).
  Embrulha em Promise.race com timeout. Fallback determinístico se Haiku falhar.
- Fetch do JSON: use no-cache (sempre puxa fresco).
- Sem libs externas exceto CDN permitidas Cowork (Chart.js, Grid.js, Mermaid).

REGISTRO
Chame mcp__cowork__create_artifact com:
- id: "btc-briefing"
- html_path: caminho do arquivo gerado
- description: "Dashboard BTC matinal — preço, vol, funding, macro, F&G, manchetes
  e síntese. Lê snapshot JSON do repo btc-forecast (recarrega a cada abertura)."
- mcp_tools: lista dos tools usados (CoinDesk e cowork)

Comece pelo Passo 0 e me avise se houver problema com o CoinDesk.
```
