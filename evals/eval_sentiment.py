"""Eval: news sentiment (src/nlp/sentiment.py), FinBERT vs the lexicon fallback.

- **Metric**: macro-F1 over the three classes (pos / neu / neg), headline only.
- **Gold set**: `evals/labels/news_100.csv`: 150 Finnhub headlines from `train` + `validation`,
  sampled by `evals/make_news_sheet.py`, labelled by a human from the view of a holder of the
  tagged ticker (D-34). Read as-is; this script never writes to it.
- **Target**: 0.80 (docs/03_eval_plan.md).
- **Why**: the classes are imbalanced (most headlines are neutral); macro-F1 weights negative and
  positive as much as neutral, so "always neutral" scores badly.

`value` is the configured backend (FinBERT when the extra is installed); the lexicon is always
reported in `details` as the baseline (D-09). `details.relevant_subset` repeats the score on the
headlines that name the tagged company (what finbert_v0 averages after its relevance filter).
Offline once the model is in the local HF cache.
"""

from __future__ import annotations

import csv
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.collectors.common import ROOT
from src.nlp.relevance import mentions
from src.nlp.sentiment import Backend, LexiconBackend, cfg, get_backend, news_text, score_texts

GOLD = ROOT / "evals" / "labels" / "news_100.csv"
TARGET = 0.80
LABELS = {"pos": 1, "neu": 0, "neg": -1}


def load_gold(path: Path = GOLD) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: sentiment gold set not available")
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    bad = [r["id"] for r in rows if r["sentiment"] not in LABELS]
    if bad:
        raise ValueError(f"unlabelled or invalid sentiment rows: {bad[:5]}")
    return rows


def macro_f1(gold: list[int], pred: list[int]) -> tuple[float, dict[int, float]]:
    per: dict[int, float] = {}
    for c in (1, 0, -1):
        tp = sum(g == c and p == c for g, p in zip(gold, pred, strict=True))
        fp = sum(g != c and p == c for g, p in zip(gold, pred, strict=True))
        fn = sum(g == c and p != c for g, p in zip(gold, pred, strict=True))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[c] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return sum(per.values()) / len(per), per


def score_backend(rows: list[dict[str, str]], backend: Backend) -> dict[str, Any]:
    gold = [LABELS[r["sentiment"]] for r in rows]
    pred = [s.label for s in score_texts([news_text(r["headline"]) for r in rows], backend)]
    f1, per = macro_f1(gold, pred)
    names = {1: "pos", 0: "neu", -1: "neg"}
    confusion = Counter(f"{names[g]}->{names[p]}" for g, p in zip(gold, pred, strict=True))
    return {
        "macro_f1": round(f1, 4),
        "f1_per_class": {names[c]: round(v, 4) for c, v in per.items()},
        "accuracy": round(sum(g == p for g, p in zip(gold, pred, strict=True)) / len(gold), 4),
        "confusion_gold_to_pred": dict(sorted(confusion.items())),
    }


def _git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return out.stdout.strip() or "unknown"


def evaluate(backend: Backend | None = None) -> dict[str, Any]:
    rows = load_gold()
    main = backend or get_backend()
    res = score_backend(rows, main)
    lex = score_backend(rows, LexiconBackend.from_config())
    # the subset finbert_v0 actually uses: headlines that name the tagged company (D-36)
    relevant = [r for r in rows if mentions(r["ticker"], r["headline"])]
    res_rel = score_backend(relevant, main)
    model = (
        f"{cfg()['finbert_model']}@{cfg()['finbert_revision'][:7]}"
        if main.method == "finbert"
        else main.method
    )
    value = res["macro_f1"]
    notes = ""
    if value < TARGET:
        notes = (
            f"macro-F1 {value} below {TARGET}; per class {res['f1_per_class']}. "
            "Explanation in docs/evals.md (sentiment)."
        )
    return {
        "component": "sentiment",
        "metric": "macro_f1",
        "value": value,
        "target": TARGET,
        "why_this_metric": "classes are imbalanced (mostly neutral); macro-F1 weights negative "
        "and positive as much as neutral",
        "gold_set": "evals/labels/news_100.csv",
        "n": len(rows),
        "split": "train+validation",
        "model": model,
        "git_sha": _git_sha(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "notes": notes,
        "details": {
            "gold_distribution": dict(Counter(r["sentiment"] for r in rows)),
            main.method: res,
            "relevant_subset": {
                "n": len(relevant),
                "gold_distribution": dict(Counter(r["sentiment"] for r in relevant)),
                main.method: res_rel,
            },
            **({} if main.method == "lexicon" else {"lexicon_baseline": lex}),
            "input": "headline only (the labelling guideline is headline only)",
        },
    }


if __name__ == "__main__":
    import json

    out = evaluate()
    path = ROOT / "evals" / "results" / "sentiment.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=2))
