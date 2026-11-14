"""Event classification of news headlines over the 8-K item taxonomy (D-10, D-39).

Two deterministic methods, both multi-label; 'none' means no label fired:

- `rule_labels`: the item codes of the tagged ticker's 8-Ks accepted within
  `[published - window_before, published + window_after]`, 9.01 dropped, items outside the
  taxonomy mapped to `other_item`. Coincidence in time, not content: a stock-move headline on an
  earnings day gets 2.02 too. A label that uses a filing accepted after the headline is only
  available at that acceptance (`rule_available_at`, D-16/D-24).
- `TfidfEventClassifier`: TF-IDF word n-grams + one-vs-rest logistic regression on the
  headline, trained on weak labels from `rule_labels` (train split only, R3). Reads the text,
  so it can tag headlines with no filing nearby and skip off-topic ones on a filing day.

The LLM classifier goes through `src/llm.py` with `rules` as its fallback (config/llm.yaml).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.collectors.common import CONFIG_DIR
from src.nlp.sentiment import prepare_text

NONE = "none"


@lru_cache(maxsize=1)
def config() -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / "extraction.yaml").read_text(encoding="utf-8"))["events"]


def labels() -> list[str]:
    return [str(x) for x in config()["labels"]]


def item_to_label(item: str) -> str | None:
    """8-K item code -> taxonomy label; None for dropped items (9.01)."""
    if item in {str(x) for x in config()["drop_items"]}:
        return None
    return item if item in labels() else "other_item"


def items_to_labels(items: Iterable[str]) -> set[str]:
    out = {lab for it in items if (lab := item_to_label(it)) is not None}
    return out or {NONE}


def _window_hits(
    published_at: datetime, filings: Sequence[tuple[datetime, Sequence[str]]]
) -> list[tuple[datetime, Sequence[str]]]:
    c = config()
    lo = published_at - timedelta(hours=float(c["window_hours_before"]))
    hi = published_at + timedelta(hours=float(c["window_hours_after"]))
    return [(at, items) for at, items in filings if lo <= at <= hi]


def rule_labels(
    published_at: datetime, filings: Sequence[tuple[datetime, Sequence[str]]]
) -> set[str]:
    """Labels from the 8-Ks (acceptance time, item codes) of the same ticker near the headline."""
    if published_at.tzinfo is None:
        raise ValueError("published_at must be timezone-aware")
    hits = _window_hits(published_at, filings)
    return items_to_labels(it for _, items in hits for it in items)


def rule_available_at(
    published_at: datetime, filings: Sequence[tuple[datetime, Sequence[str]]]
) -> datetime:
    """When the rule label is observable: the latest of the headline and the filings it used."""
    hits = [
        at for at, items in _window_hits(published_at, filings) if items_to_labels(items) != {NONE}
    ]
    return max([published_at, *hits])


class TfidfEventClassifier:
    """TF-IDF + one-vs-rest logistic regression; labels with P >= threshold, else {'none'}."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or config()
        t = self.cfg["tfidf"]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, int(t["ngram_max"])),
            min_df=int(t["min_df"]),
            max_features=int(t["max_features"]),
            sublinear_tf=bool(t["sublinear_tf"]),
            lowercase=True,
        )
        self.models: dict[str, LogisticRegression] = {}
        self.constant: dict[str, float] = {}  # labels with a single class in training
        self.label_names = [str(x) for x in self.cfg["labels"]]

    def fit(self, texts: Sequence[str], targets: Sequence[set[str]]) -> TfidfEventClassifier:
        x = self.vectorizer.fit_transform([prepare_text(t) for t in texts])
        lr = self.cfg["logreg"]
        for lab in self.label_names:
            y = np.array([lab in t for t in targets], dtype=int)
            if y.min() == y.max():
                self.constant[lab] = float(y[0]) if len(y) else 0.0
                continue
            m = LogisticRegression(
                C=float(lr["C"]),
                class_weight=lr["class_weight"],
                max_iter=int(lr["max_iter"]),
            )
            self.models[lab] = m.fit(x, y)
        return self

    def predict_proba(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        x = self.vectorizer.transform([prepare_text(t) for t in texts])
        out = {lab: m.predict_proba(x)[:, 1] for lab, m in self.models.items()}
        for lab, p in self.constant.items():
            out[lab] = np.full(x.shape[0], p)
        return out

    def predict(self, texts: Sequence[str]) -> list[set[str]]:
        probs = self.predict_proba(texts)
        thr = float(self.cfg["threshold"])
        preds = []
        for i in range(len(texts)):
            labs = {lab for lab in self.label_names if probs[lab][i] >= thr}
            preds.append(labs or {NONE})
        return preds
