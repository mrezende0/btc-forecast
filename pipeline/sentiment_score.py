"""Sentiment scorer uniforme via FinBERT (ProsusAI/finbert).

Função pura: recebe lista de textos, retorna scores em [-1, +1].

Modelo é carregado lazy (primeira chamada) e cacheado em módulo. ~500MB no primeiro
download.

Override via env:
  SENTIMENT_MODEL=ElKulako/cryptobert    # para crypto social media
  SENTIMENT_DEVICE=cuda                  # se houver GPU
"""
from __future__ import annotations

import os
from typing import Iterable

_MODEL = None
_TOKENIZER = None
_DEVICE = None

DEFAULT_MODEL = os.environ.get("SENTIMENT_MODEL", "ProsusAI/finbert")
MAX_LEN = 256  # headlines + lead = curto, 256 tokens cobre


def _load() -> None:
    global _MODEL, _TOKENIZER, _DEVICE
    if _MODEL is not None:
        return
    # Import dentro pra não exigir torch/transformers em jobs que só ingerem.
    import torch  # type: ignore
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

    _DEVICE = os.environ.get("SENTIMENT_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    _TOKENIZER = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    _MODEL = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL)
    _MODEL.eval()
    _MODEL.to(_DEVICE)


def score(texts: Iterable[str], batch_size: int = 32) -> list[float]:
    """Score uniforme em [-1, +1].

    FinBERT retorna 3 logits: [positive, negative, neutral].
    Convertemos pra escalar via: P(pos) - P(neg). Neutro contribui zero.
    """
    items = [t if isinstance(t, str) and t.strip() else "" for t in texts]
    if not items:
        return []

    _load()
    import torch  # type: ignore

    out: list[float] = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        # Substitui vazios por placeholder neutro pra não quebrar tokenizer
        clean = [b if b else "neutral" for b in batch]
        enc = _TOKENIZER(
            clean,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(_DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = _MODEL(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        # ProsusAI/finbert label order: 0=positive, 1=negative, 2=neutral
        for j, p in enumerate(probs):
            if not batch[j]:
                out.append(0.0)
                continue
            out.append(float(p[0] - p[1]))  # P(pos) - P(neg)
    return out
