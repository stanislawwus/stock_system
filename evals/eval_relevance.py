"""Eval: near-duplicate removal (src/nlp/relevance.py), plus descriptive relevance shares.

- **Metric**: `dedup_f1` = F1 of "this item is dropped as a near-duplicate", precision and recall
  reported separately in `details` (eval plan targets: P >= 0.90, R >= 0.80).
- **Gold set**: derived. 30 ticker-days of the `train` split (watchlist, fixed seed) with at
  least 10 relevant headlines with distinct dedup keys; 10 of them per day at their real times
  are the originals, and each gets one planted copy 5-600 minutes later made by a
  syndication-style edit (wire prefix, source suffix, upper case, punctuation, one dropped word,
  one swapped word, two dropped words + suffix, clauses reordered). Gold: exactly the planted
  copies are duplicates. The other same-day headlines about the same company are the negatives,
  so precision measures whether different stories about one company on one day are kept apart.
  Caveat: two originals can themselves be real near-duplicates (different dedup keys); such a
  drop counts as a false positive here, and the first ones are listed in `fp_examples`.
- **Target**: 0.85 (harmonic mean of P = 0.90 and R = 0.80).
- **Why**: dropped originals bias the daily sentiment mean (precision); missed copies let one
  syndicated story count several times (recall).

Reads the database read-only, `train` rows only (R3). No model, no network.
"""

from __future__ import annotations

import random
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src import db
from src.collectors.common import ROOT, load_periods, watchlist_tickers
from src.marketview.clock import cutoff_for
from src.nlp.clean import dedup_key
from src.nlp.relevance import config, dedup, is_relevant
from src.schemas import NewsItem

N_DAYS = 30
PER_DAY = 10
SEED = 20261114
TARGET = 0.85
EDITS = (
    "wire_prefix",
    "source_suffix",
    "upper",
    "punctuation",
    "drop_word",
    "swap_word",
    "drop_two_suffix",
    "reorder",
)
SWAPS = {
    "rises": "climbs",
    "falls": "drops",
    "stock": "shares",
    "shares": "stock",
    "says": "said",
    "to": "will",
    "and": "&",
}


def plant(text: str, edit: str, rng: random.Random) -> str:
    words = text.split()
    if edit == "wire_prefix":
        return f"UPDATE 1-{text}"
    if edit == "source_suffix":
        return f"{text} - {rng.choice(['Reuters', 'Benzinga', 'MarketWatch', 'Bloomberg'])}"
    if edit == "upper":
        return text.upper()
    if edit == "punctuation":
        return text.replace(",", "").replace(":", " -").replace("'", "") + "!"
    if edit == "drop_word" and len(words) > 4:
        del words[rng.randrange(1, len(words))]
        return " ".join(words)
    if edit == "drop_two_suffix" and len(words) > 6:
        for _ in range(2):
            del words[rng.randrange(1, len(words))]
        return " ".join(words) + " - Yahoo Finance"
    if edit == "reorder":
        for sep in (": ", "; ", ", "):
            if sep in text:
                head, tail = text.split(sep, 1)
                return f"{tail}{sep}{head}"
        return " ".join(words[len(words) // 2 :] + words[: len(words) // 2])
    if edit == "swap_word":
        for i, w in enumerate(words):
            if w.lower() in SWAPS:
                words[i] = SWAPS[w.lower()]
                return " ".join(words)
        return f"{text} (update)"
    return f"{text} - Reuters"


def _git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return out.stdout.strip() or "unknown"


def load_train_news(db_path: Path | None = None) -> list[NewsItem]:
    (train,) = load_periods(["train"])
    lo, hi = cutoff_for(train.start), cutoff_for(train.end)
    con = db.connect(db_path)
    try:
        rows = con.execute(
            "SELECT id, ticker, published_at, headline, summary, url, source, available_at "
            "FROM news WHERE list_contains(?, ticker) AND available_at > ? AND available_at <= ? "
            "ORDER BY available_at, id",
            [watchlist_tickers(), lo, hi],
        ).fetchall()
    finally:
        con.close()
    cols = ("id", "ticker", "published_at", "headline", "summary", "url", "source", "available_at")
    return [NewsItem(**dict(zip(cols, r, strict=True))) for r in rows]


def build_gold(news: list[NewsItem]) -> tuple[list[NewsItem], set[str], dict[str, str]]:
    """(items, ids of planted copies, edit per planted id)."""
    rng = random.Random(SEED)
    days: dict[tuple[str, object], list[NewsItem]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for n in news:
        k = (n.ticker, dedup_key(n.headline))
        if is_relevant(n) and k not in seen:
            seen.add(k)
            days[(n.ticker, n.available_at.date())].append(n)
    eligible = sorted(k for k, v in days.items() if len(v) >= PER_DAY)
    items: list[NewsItem] = []
    planted: set[str] = set()
    edits: dict[str, str] = {}
    for key in rng.sample(eligible, min(N_DAYS, len(eligible))):
        for n in sorted(rng.sample(days[key], PER_DAY), key=lambda x: (x.available_at, x.id)):
            items.append(n)
            edit = EDITS[len(planted) % len(EDITS)]
            pid = f"planted:{n.id}"
            at = n.available_at + timedelta(minutes=rng.randint(5, 600))
            update = {
                "id": pid,
                "headline": plant(n.headline, edit, rng),
                "available_at": at,
                "published_at": at,
            }
            items.append(n.model_copy(update=update))
            planted.add(pid)
            edits[pid] = edit
    return items, planted, edits


def evaluate(db_path: Path | None = None) -> dict[str, Any]:
    news = load_train_news(db_path)
    items, planted, edits = build_gold(news)
    res = dedup(items)
    dropped = set(res.dropped)
    tp = len(dropped & planted)
    fp = len(dropped - planted)
    fn = len(planted - dropped)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    by_edit = Counter(edits[i] for i in planted)
    caught = Counter(edits[i] for i in planted & dropped)
    by_id = {n.id: n for n in items}
    fp_examples = [
        [by_id[i].headline, by_id[res.dropped[i]].headline] for i in sorted(dropped - planted)[:5]
    ]
    rel = Counter((n.ticker, is_relevant(n)) for n in news)
    shares = {
        t: round(rel[(t, True)] / max(1, rel[(t, True)] + rel[(t, False)]), 3)
        for t in watchlist_tickers()
    }
    notes = ""
    if f1 < TARGET:
        notes = f"precision {p:.2f}, recall {r:.2f}; see recall_by_edit and fp_examples."
    cfg = config()["dedup"]
    return {
        "component": "relevance",
        "metric": "dedup_f1",
        "value": round(f1, 4),
        "target": TARGET,
        "why_this_metric": "dropped originals bias the daily sentiment mean (precision), missed "
        "syndicated copies count one story several times (recall); F1 balances both",
        "gold_set": "derived: 30 train ticker-days x 10 relevant distinct watchlist headlines "
        "at real times + one planted syndication-style copy each",
        "n": len(items),
        "split": "train",
        "model": f"minhash(char{cfg['shingle_chars']}, {cfg['num_perm']} perms, "
        f"thr {cfg['threshold']})",
        "git_sha": _git_sha(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "notes": notes,
        "details": {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "precision_target": 0.90,
            "recall_target": 0.80,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "recall_by_edit": {e: round(caught[e] / by_edit[e], 3) for e in sorted(by_edit)},
            "fp_examples": fp_examples,
            "relevant_share_train_by_ticker": shares,
            "relevant_share_train_all": round(
                sum(v for (_, ok), v in rel.items() if ok) / max(1, len(news)), 4
            ),
        },
    }


if __name__ == "__main__":
    import json

    out = evaluate()
    path = ROOT / "evals" / "results" / "relevance.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: out[k] for k in ("metric", "value", "target")}), out["details"])
