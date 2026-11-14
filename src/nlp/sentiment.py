"""Sentiment of financial text: FinBERT (ProsusAI/finbert) batches, lexicon fallback (D-09, D-36).

`prepare_text` runs first on every input: HTML stripped, 8-K item headings removed (they name the
event type, D-34), whitespace normalised, truncated to `max_chars` (untrusted text, R6).
FinBERT is a classifier, not an instruction follower, so text inside the input cannot change the
output schema; the injection test in tests/nlp/test_sentiment.py pins that for both backends.

Score = P(positive) - P(negative) in [-1, 1]; label = argmax mapped to 1 / 0 / -1.
FinBERT needs the optional extra (`uv sync --extra finbert`); without it `get_backend()` returns
the lexicon and logs the fallback. Batch results for news are cached in the DuckDB `sentiment`
table by `src/nlp/features.py` (`make features`); this module does no I/O apart from loading the
model and the config.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any, Protocol

from src.nlp.clean import normalize_text, strip_8k_item_headers
from src.nlp.relevance import config as nlp_config
from src.schemas import Sentiment
from src.textutil import strip_html

log = logging.getLogger(__name__)
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


@dataclass(frozen=True)
class Scored:
    label: int  # -1 | 0 | 1
    score: float  # P(pos) - P(neg), or the lexicon balance
    probs: tuple[float, float, float] | None  # (neg, neu, pos) for FinBERT


class Backend(Protocol):
    method: str  # value stored in sentiment.method

    def predict(self, texts: Sequence[str]) -> list[Scored]: ...


def cfg() -> dict[str, Any]:
    return nlp_config()["sentiment"]


def prepare_text(text: str) -> str:
    """Untrusted text -> model input: no HTML, no 8-K item headings, bounded length."""
    plain = strip_html(text, keep_lines=True)
    return normalize_text(strip_8k_item_headers(plain))[: int(cfg()["max_chars"])]


def news_text(headline: str, summary: str = "") -> str:
    mode = cfg()["news_text"]
    if mode == "headline":
        return headline
    if mode == "headline_summary":
        return f"{headline}. {summary}" if summary else headline
    raise ValueError(f"unknown sentiment.news_text {mode!r}")


# --------------------------------------------------------------------------- backends


class LexiconBackend:
    """Deterministic fallback: (#pos - #neg) / (#pos + #neg) over exact lower-cased words."""

    method = "lexicon"

    def __init__(self, positive: Sequence[str], negative: Sequence[str]) -> None:
        self.pos = frozenset(w.lower() for w in positive)
        self.neg = frozenset(w.lower() for w in negative)

    @classmethod
    def from_config(cls) -> LexiconBackend:
        lex = nlp_config()["lexicon"]
        return cls(lex["positive"], lex["negative"])

    def predict(self, texts: Sequence[str]) -> list[Scored]:
        out = []
        for t in texts:
            words = _WORD_RE.findall(t.lower())
            p = sum(w in self.pos for w in words)
            n = sum(w in self.neg for w in words)
            score = (p - n) / (p + n) if p + n else 0.0
            out.append(Scored(label=(score > 0) - (score < 0), score=score, probs=None))
        return out


class FinBertBackend:
    """ProsusAI/finbert via transformers, CPU or GPU, batched, no gradient."""

    method = "finbert"

    def __init__(self, model_name: str, revision: str, batch_size: int, max_tokens: int) -> None:
        import torch  # optional extra: ImportError is handled by get_backend()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, revision=revision
        )
        self.model.eval()
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        labels = {i: str(lab).lower() for i, lab in self.model.config.id2label.items()}
        self._idx = {name: i for i, name in labels.items()}
        if set(self._idx) != {"positive", "negative", "neutral"}:
            raise RuntimeError(f"unexpected FinBERT labels {labels}")

    @classmethod
    def from_config(cls) -> FinBertBackend:
        c = cfg()
        return cls(
            c["finbert_model"], c["finbert_revision"], int(c["batch_size"]), int(c["max_tokens"])
        )

    def predict(self, texts: Sequence[str]) -> list[Scored]:
        out: list[Scored] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            )
            with self._torch.no_grad():
                probs = self._torch.softmax(self.model(**enc).logits, dim=-1).tolist()
            for p in probs:
                neg = p[self._idx["negative"]]
                neu = p[self._idx["neutral"]]
                pos = p[self._idx["positive"]]
                label = max(((pos, 1), (neu, 0), (neg, -1)))[1]
                out.append(Scored(label=label, score=pos - neg, probs=(neg, neu, pos)))
        return out


_override: Backend | None = None


def set_backend(backend: Backend | None) -> None:
    """Tests and evals inject a backend; None restores the configured one."""
    global _override
    _override = backend


@lru_cache(maxsize=2)
def _configured(name: str) -> Backend:
    if name in ("auto", "finbert"):
        try:
            return FinBertBackend.from_config()
        except ImportError:
            if name == "finbert":
                raise
            log.warning(
                "sentiment: FinBERT extra not installed (uv sync --extra finbert); "
                "falling back to the lexicon (D-09)"
            )
    elif name != "lexicon":
        raise ValueError(f"unknown sentiment backend {name!r}")
    return LexiconBackend.from_config()


def get_backend(name: str | None = None) -> Backend:
    if _override is not None and name is None:
        return _override
    return _configured(name or cfg()["backend"])


# --------------------------------------------------------------------------- service


def score_texts(texts: Sequence[str], backend: Backend | None = None) -> list[Scored]:
    """Batch inference on prepared texts (one result per input, same order)."""
    b = backend or get_backend()
    return b.predict([prepare_text(t) for t in texts])


def analyze_text(text: str, ticker: str | None, at: datetime) -> Sentiment:
    """One text -> Sentiment record (POST /analyze). `at` = when the text was submitted, which
    is the record's available_at (D-16)."""
    b = get_backend()
    (s,) = score_texts([text], b)
    ref = "analyze:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return Sentiment(
        available_at=at,
        source_ref=ref,
        ticker=(ticker or "").upper(),
        label=s.label,
        score=max(-1.0, min(1.0, s.score)),
        method=b.method,
    )
