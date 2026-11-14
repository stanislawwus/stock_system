"""Eval: news sentiment (src/nlp/sentiment.py), FinBERT vs the lexicon fallback.

- **Metric**: macro-F1 over the three classes (pos / neu / neg), headline only.
- **Gold set**: `evals/labels/news_100.csv`: 150 Finnhub headlines from `train` + `validation`,
  sampled by `evals/make_news_sheet.py`, labelled by a human from the view of a holder of the
  tagged ticker (D-34). Read as-is; this script never writes to it.
- **Target**: 0.80 (docs/03_eval_plan.md).
- **Why**: the classes are imbalanced (most headlines are neutral); macro-F1 weights negative and
  positive as much as neutral, so "always neutral" scores badly.

`value` is the configured backend (FinBERT when the extra is installed); the lexicon is always
reported in `details` as the baseline (D-09). Every score carries per-class precision/recall/F1,
the confusion matrix (gold rows, predicted columns) and a 95 % i.i.d. bootstrap CI of macro-F1
(`config/evals.yaml`).

Second gold set (`details.twitter_financial`): the validation split of the public
`zeroshot/twitter-financial-news-sentiment` (2,388 finance tweets, bearish/bullish/neutral),
pinned by revision and sha256 in `config/evals.yaml`, cached in `data/raw/hf/` (R15). FinBERT was
fine-tuned on Financial PhraseBank, so a PhraseBank score would flatter it; this set is
independent of both FinBERT's training data and our own labelling guideline (D-39).
`details.relevant_subset` repeats the score on the headlines that name the tagged company (what
finbert_v0 averages after its relevance filter). Offline once the model is in the local HF cache
and the Twitter file in data/raw.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from evals.metrics import bootstrap_ci, classification_report, eval_config, git_sha, macro_f1
from src.collectors.common import ROOT
from src.nlp.relevance import mentions
from src.nlp.sentiment import Backend, LexiconBackend, cfg, get_backend, news_text, score_texts

GOLD = ROOT / "evals" / "labels" / "news_100.csv"
TARGET = 0.80
LABELS = {"pos": 1, "neu": 0, "neg": -1}
CLASSES = ("pos", "neu", "neg")
_URL_RE = re.compile(r"https?://\S+")


def load_gold(path: Path = GOLD) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: sentiment gold set not available")
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    bad = [r["id"] for r in rows if r["sentiment"] not in LABELS]
    if bad:
        raise ValueError(f"unlabelled or invalid sentiment rows: {bad[:5]}")
    return rows


def score_backend(rows: list[dict[str, str]], backend: Backend) -> dict[str, Any]:
    """Report for `rows` with keys `text` (model input) and `sentiment` (pos/neu/neg)."""
    gold = [r["sentiment"] for r in rows]
    names = {1: "pos", 0: "neu", -1: "neg"}
    pred = [names[s.label] for s in score_texts([r["text"] for r in rows], backend)]
    rep = classification_report(gold, pred, CLASSES)
    g, p = np.asarray(gold), np.asarray(pred)
    rep["macro_f1_ci95"] = bootstrap_ci(len(g), lambda i: macro_f1(g[i], p[i], CLASSES))
    rep["recall_per_class"] = {c: rep["per_class"][c]["recall"] for c in CLASSES}
    return rep


def _with_text(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{**r, "text": news_text(r["headline"])} for r in rows]


def load_twitter() -> list[dict[str, str]]:
    """The pinned Twitter financial sentiment file; downloaded once into data/raw (R15)."""
    c = eval_config()["twitter_financial"]
    path = ROOT / "data" / "raw" / "hf" / c["name"] / c["file"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        resp = httpx.get(c["url"], follow_redirects=True, timeout=60.0)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != c["sha256"]:
        raise ValueError(f"{path} sha256 {digest} != pinned {c['sha256']}")
    label_map = {int(k): v for k, v in c["label_map"].items()}
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        text = _URL_RE.sub("", r["text"]).strip() if c["strip_urls"] else r["text"]
        out.append({"text": text, "sentiment": label_map[int(r["label"])]})
    return out


def evaluate(backend: Backend | None = None) -> dict[str, Any]:
    rows = _with_text(load_gold())
    main = backend or get_backend()
    res = score_backend(rows, main)
    lex_backend = LexiconBackend.from_config()
    lex = score_backend(rows, lex_backend)
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
            f"macro-F1 {value} (95% CI {res['macro_f1_ci95']}) below {TARGET}; recall per class "
            f"{res['recall_per_class']}. "
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
        "git_sha": git_sha(),
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
            "twitter_financial": _twitter_block(main, lex_backend),
            "input": "headline only (the labelling guideline is headline only)",
        },
    }


def _twitter_block(main: Backend, lex: Backend) -> dict[str, Any]:
    c = eval_config()["twitter_financial"]
    try:
        tw = load_twitter()
    except (httpx.HTTPError, OSError) as exc:  # offline clean clone: report, do not fail
        return {"skipped": f"{type(exc).__name__}: {exc}"}
    out: dict[str, Any] = {
        "source": f"hf:zeroshot/twitter-financial-news-sentiment@{c['revision'][:7]}/{c['file']}",
        "n": len(tw),
        "gold_distribution": dict(Counter(r["sentiment"] for r in tw)),
        main.method: score_backend(tw, main),
    }
    if main.method != "lexicon":
        out["lexicon_baseline"] = score_backend(tw, lex)
    return out


if __name__ == "__main__":
    import json

    out = evaluate()
    path = ROOT / "evals" / "results" / "sentiment.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=2))
