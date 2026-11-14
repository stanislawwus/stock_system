"""Eval: news event classification (src/nlp/events.py), (a) 8-K item rules vs (b) TF-IDF + LR.

- **Metric**: macro-F1 over the event labels with gold support (multi-label; 'none' counts as a
  label, 9.01 dropped), micro-F1, exact set match and per-label precision/recall reported too,
  each with a 95 % i.i.d. bootstrap CI of the macro/micro F1.
- **Gold set**: the `events` column of `evals/labels/news_100.csv` (150 human-labelled
  headlines, train + validation).
- **Training data for (b)**: every headline of the 77-name universe in the `train` split that
  names its tagged company, weak-labelled by (a); gold ids are excluded. Nothing is fitted on
  validation or on the gold set (R3).
- **Target**: 0.75 macro-F1 (docs/03_eval_plan.md).
- **Why**: event classes are rare and unequal; accuracy would reward predicting 'none'.

`value` is (b), the text model; (a) is the baseline and the weak labeller. Caveat: news_100 was
sampled stratified by "an 8-K of the tagged ticker within +/- 24 h" (D-34) with the same window
the rule uses, so (a) sees the sampling variable; that inflates its recall on this set.
"""

from __future__ import annotations

import csv
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evals.metrics import bootstrap_ci, git_sha, multilabel_report
from src import db
from src.collectors.common import ROOT, load_periods
from src.marketview.clock import cutoff_for
from src.nlp.events import NONE, TfidfEventClassifier, config, labels, rule_labels
from src.nlp.relevance import mentions

GOLD = ROOT / "evals" / "labels" / "news_100.csv"
TARGET = 0.75


def load_gold(path: Path = GOLD) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: events gold set not available")
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    dropped = {str(x) for x in config()["drop_items"]}
    for r in rows:
        labs = {x.strip() for x in r["events"].split("|") if x.strip()} - dropped
        r["gold"] = labs or {NONE}
        r["published_at"] = datetime.fromisoformat(r["published_at"].replace("Z", "+00:00"))
    return rows


def load_filings(con: Any) -> dict[str, list[tuple[datetime, list[str]]]]:
    rows = con.execute(
        "SELECT ticker, accepted_at, items FROM filings WHERE form = '8-K' ORDER BY accepted_at"
    ).fetchall()
    out: dict[str, list[tuple[datetime, list[str]]]] = {}
    for ticker, at, items in rows:
        out.setdefault(ticker, []).append((at.astimezone(UTC), list(items)))
    return out


def load_train_headlines(con: Any, exclude: set[str]) -> pd.DataFrame:
    (train,) = load_periods(["train"])
    lo, hi = cutoff_for(train.start), cutoff_for(train.end)
    df = con.execute(
        "SELECT id, ticker, published_at, headline FROM news "
        "WHERE available_at > ? AND available_at <= ?",
        [lo, hi],
    ).df()
    df = df[~df["id"].isin(exclude)]
    if config()["train_relevant_only"]:
        df = df[[mentions(t, h) for t, h in zip(df["ticker"], df["headline"], strict=True)]]
    return df.drop_duplicates(["ticker", "headline"]).reset_index(drop=True)


def weak_labels(df: pd.DataFrame, filings: dict[str, list]) -> list[set[str]]:
    out = []
    for t, at in zip(df["ticker"], df["published_at"], strict=True):
        ts = pd.Timestamp(at).tz_convert("UTC").to_pydatetime()
        out.append(rule_labels(ts, filings.get(t, [])))
    return out


def _report(gold: list[set[str]], pred: list[set[str]], space: list[str]) -> dict[str, Any]:
    rep = multilabel_report(gold, pred, space)
    n = len(gold)
    g, p = np.array(gold, dtype=object), np.array(pred, dtype=object)
    rep["macro_f1_ci95"] = bootstrap_ci(
        n, lambda i: multilabel_report(list(g[i]), list(p[i]), space)["macro_f1"]
    )
    rep["micro_f1_ci95"] = bootstrap_ci(
        n, lambda i: multilabel_report(list(g[i]), list(p[i]), space)["micro_f1"]
    )
    return rep


def evaluate(db_path: Path | None = None) -> dict[str, Any]:
    gold_rows = load_gold()
    space = [*labels(), NONE]
    con = db.connect(db_path)
    try:
        filings = load_filings(con)
        train = load_train_headlines(con, exclude={r["id"] for r in gold_rows})
    finally:
        con.close()
    gold = [r["gold"] for r in gold_rows]
    rules = [rule_labels(r["published_at"], filings.get(r["ticker"], [])) for r in gold_rows]
    weak = weak_labels(train, filings)
    clf = TfidfEventClassifier().fit(train["headline"].tolist(), weak)
    tfidf = clf.predict([r["headline"] for r in gold_rows])
    rep_a, rep_b = _report(gold, rules, space), _report(gold, tfidf, space)
    # rows sampled with no 8-K nearby (stratum 'none'): where the rule cannot see the answer
    strata = {}
    with (GOLD.parent / "news_100.strata.csv").open(encoding="utf-8", newline="") as f:
        strata = {r["id"]: r["stratum"] for r in csv.DictReader(f)}
    no8k = [i for i, r in enumerate(gold_rows) if strata.get(r["id"]) == "none"]
    value = rep_b["macro_f1"]
    notes = ""
    if value < TARGET:
        notes = (
            f"TF-IDF+LR macro-F1 {value} (CI {rep_b['macro_f1_ci95']}) and rules "
            f"{rep_a['macro_f1']} (CI {rep_a['macro_f1_ci95']}) below {TARGET}. "
            "Explanation in docs/evals.md (events)."
        )
    weak_dist = Counter(lab for s in weak for lab in s)
    return {
        "component": "events",
        "metric": "macro_f1",
        "value": value,
        "target": TARGET,
        "why_this_metric": "event classes are rare and unequal; macro-F1 over labels with gold "
        "support weights each event type equally, accuracy would reward predicting 'none'",
        "gold_set": "evals/labels/news_100.csv (events column)",
        "n": len(gold_rows),
        "split": "train (fit, weak labels) / train+validation (gold)",
        "model": "tfidf+logreg (weak labels from 8-K items)",
        "git_sha": git_sha(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "notes": notes,
        "details": {
            "label_space": space,
            "gold_distribution": dict(Counter(lab for s in gold for lab in s)),
            "a_rules_8k_items": rep_a,
            "b_tfidf_logreg": rep_b,
            "no_8k_stratum": {
                "n": len(no8k),
                "a_rules_exact_set_match": round(
                    float(np.mean([rules[i] == gold[i] for i in no8k])), 4
                ),
                "b_tfidf_exact_set_match": round(
                    float(np.mean([tfidf[i] == gold[i] for i in no8k])), 4
                ),
            },
            "train": {
                "n_headlines": len(train),
                "weak_label_distribution": dict(weak_dist.most_common()),
                "constant_labels": sorted(clf.constant),
            },
            "config": {k: config()[k] for k in ("tfidf", "logreg", "threshold")},
            "caveat": "news_100 strata use the same +/- 24 h 8-K window as the rule (D-34)",
        },
    }


if __name__ == "__main__":
    import json

    out = evaluate()
    path = ROOT / "evals" / "results" / "events.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in out.items() if k != "details"}, indent=2))
