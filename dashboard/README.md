# Dashboard — BTC Briefing

Dashboard pessoal de leitura matinal do mercado BTC, rodando no Cowork (app
desktop do Claude). Consome dados do nosso pipeline + CoinDesk MCP ao vivo.

## Setup (5 min, uma vez)

### 1. Cowork instalado e logado
Baixar em claude.ai/download, login com mesma conta Claude, modo Cowork ativo.

### 2. CoinDesk MCP conectado
Settings → Connectors → busca "CoinDesk" → Connect → cola API key.
(API key gratuita em developers.coindesk.com)

### 3. Cole o prompt master
Abre `PROMPT_MASTER.md` nesta pasta, copia o bloco do prompt, cola numa nova
conversa do Cowork, Enter. Cowork constrói o artifact em ~30s e registra.

### 4. Marca o artifact
Aparece "BTC Briefing" na barra lateral de Artifacts. Toda manhã, abre + Reload.

## Como funciona

```
GitHub Actions (ingest_daily, 6h UTC)
  └─ pipeline.dashboard_state → data/dashboard_state.json (commitado)

Cowork (manhã)
  ├─ Fetch raw.githubusercontent.com/.../dashboard_state.json   ← dados nossos
  └─ CoinDesk MCP fetch_news                                    ← manchetes live
       ↓
     Renderiza HTML com 9 seções + síntese Haiku
```

Snapshot JSON tem: preço/vol/funding/macro/F&G/sentiment (quando GDELT terminar).
Manchetes vêm sempre frescas do CoinDesk.

## Seções

1. Header — preço atual + 24h
2. Preço e estrutura (24h/7d/30d/90d + range 30d)
3. Volatilidade (1d vs 1w vs 30d, regime)
4. Funding (z-score, leitura tática)
5. Macro D-1 (DXY/VIX/SPX)
6. Sentimento (F&G + sentiment news quando disponível)
7. Manchetes 48h (CoinDesk live, com sentiment pills)
8. "O que olharia hoje" (Haiku síntese + fallback determinístico)
9. Diagnóstico (idade snapshot, volumes coletados)

## Quando der errado

**Dashboard mostra "sentiment_news indisponível"**
GDELT backfill ainda não rodou. Esperado nos primeiros dias do projeto.

**Manchetes do CoinDesk não carregam**
Conector pode ter desconectado. Settings → Connectors → CoinDesk → Reconnect.

**JSON antigo (generated_at > 24h)**
Workflow `ingest_daily` não rodou hoje. Conferir GitHub Actions → Ingest daily.

**UUIDs de MCP mudaram entre sessões**
Pede pro Cowork: "verifique conectores e atualize o artifact BTC Briefing"

## Próximo nível (futuro)

- Adicionar seção "Sinal modelo v2" lendo predictions.parquet (quando paper trading começar)
- Histórico de sinais (acerto/erro nos últimos 30d)
- Alerta visual quando funding_z > 2 ou F&G < 20
