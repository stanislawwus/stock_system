"""Relevance (symbol/alias mention) and MinHash near-duplicate removal (D-36)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.nlp.relevance import (
    MinHasher,
    dedup,
    is_relevant,
    jaccard_estimate,
    mentions,
    relevant_unique,
)
from src.schemas import NewsItem

T0 = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


def item(i: str, headline: str, hours: float = 0.0, ticker: str = "AAPL", summary: str = ""):
    at = T0 + timedelta(hours=hours)
    return NewsItem(
        id=i,
        ticker=ticker,
        published_at=at,
        available_at=at,
        headline=headline,
        summary=summary,
        url=f"https://example.com/{i}",
        source="test",
    )


@pytest.mark.parametrize(
    ("ticker", "text", "expected"),
    [
        ("JNJ", "Johnson &amp; Johnson raises outlook", True),  # HTML entity from the feed
        ("JNJ", "J&J talc verdict", True),
        ("JNJ", "ISRG Stock: What Does Its Valuation Imply?", False),
        ("MCD", "McDonald’s same-store sales rise", True),  # typographic apostrophe
        ("AAPL", "AAPL rallies into the close", True),
        ("AAPL", "Pineapple prices jump", False),  # alias inside a word is not a mention
        ("BA", "BA shares fall", False),  # ambiguous bare symbol needs a marker
        ("BA", "Boeing (BA) delivers 40 jets", True),
        ("BA", "$BA up 3%", True),
        ("T", "AT&T beats on subscribers", True),
        ("T", "T stock: is it a buy?", False),
        ("LOW", "LOW RATES BOOST HOMEBUILDERS", False),  # all-caps headline, ambiguous symbol
        ("LOW", "Lowe's cuts guidance", True),
        ("GS", "Goldman, CrowdStrike jump", True),
        # regression: the conference name is not a JPM mention (docs/evals.md relevance)
        ("JPM", "Gilead to Present at the 44th Annual J.P. Morgan Healthcare Conference", False),
        ("JPM", "Amgen at JPMorgan Healthcare Conference: pipeline update", False),
        ("JPM", "Jamie Dimon speaks at the J.P. Morgan Healthcare Conference", True),
        ("JPM", "J.P. Morgan raises S&P 500 target", True),
    ],
)
def test_mentions(ticker: str, text: str, expected: bool) -> None:
    assert mentions(ticker, text) is expected


def test_relevance_reads_the_summary_too() -> None:
    assert is_relevant(
        item("a", "Big pharma stocks to watch", ticker="JNJ", summary="Johnson & Johnson leads")
    )
    assert not is_relevant(item("b", "Big pharma stocks to watch", ticker="JNJ", summary="Pfizer"))


def test_minhash_estimates_jaccard() -> None:
    mh = MinHasher(num_perm=256, shingle_chars=4, seed=1)
    a = "Apple beats estimates as iPhone sales climb"
    assert jaccard_estimate(mh.signature(a), mh.signature(a)) == 1.0
    # true Jaccard of the 4-shingle sets, then the estimate within MinHash noise
    b = "Apple misses estimates as Mac sales fall"
    sa, sb = mh.shingles(a), mh.shingles(b)
    true_j = len(sa & sb) / len(sa | sb)
    est = jaccard_estimate(mh.signature(a), mh.signature(b))
    assert abs(est - true_j) < 0.1


BASE = [
    "Apple beats estimates as iPhone sales climb",
    "Apple to open new campus in Texas",
    "Apple faces EU antitrust fine over App Store rules",
    "Apple supplier Foxconn reports record revenue",
    "Apple Vision Pro shipments slow, analysts say",
]


def planted(text: str, k: int) -> str:
    """Syndication-style edits: wire prefix, source suffix, case, punctuation."""
    return [
        f"UPDATE 1-{text}",
        f"{text} - Reuters",
        text.upper(),
        text.replace(" ", "  ").replace(",", "") + "!",
        f"{text} | Benzinga",
    ][k % 5]


def test_planted_duplicates_are_removed_and_originals_kept() -> None:
    items = [item(f"o{i}", h, hours=i) for i, h in enumerate(BASE)]
    dups = [item(f"d{i}", planted(h, i), hours=i + 0.5) for i, h in enumerate(BASE)]
    res = dedup(items + dups)
    assert {n.id for n in res.kept} == {f"o{i}" for i in range(len(BASE))}
    assert res.dropped == {f"d{i}": f"o{i}" for i in range(len(BASE))}


def test_the_earliest_copy_survives_whatever_the_input_order() -> None:
    late = item("late", "Apple beats estimates as iPhone sales climb", hours=3)
    early = item("early", "UPDATE 1-Apple beats estimates as iPhone sales climb", hours=1)
    assert [n.id for n in dedup([late, early]).kept] == ["early"]


def test_copies_outside_the_window_or_of_another_ticker_are_kept() -> None:
    a = item("a", BASE[0])
    far = item("far", planted(BASE[0], 1), hours=48)
    other = item("other", BASE[0], hours=1, ticker="MSFT")
    assert {n.id for n in dedup([a, far, other], window_hours=24).kept} == {"a", "far", "other"}


def test_dedup_is_causal_a_prefix_gives_the_same_decisions() -> None:
    items = [item(f"o{i}", h, hours=i) for i, h in enumerate(BASE)]
    items += [item(f"d{i}", planted(h, i), hours=i + 0.5) for i, h in enumerate(BASE)]
    full = dedup(items)
    cut = T0 + timedelta(hours=2.7)
    prefix = dedup([n for n in items if n.available_at <= cut])
    assert {n.id for n in prefix.kept} == {n.id for n in full.kept if n.available_at <= cut}


def test_relevant_unique_filters_before_dedup() -> None:
    off = item("off", "Stocks to watch: Apple", hours=0, ticker="MSFT")
    on = item("on", "Microsoft beats estimates", hours=1, ticker="MSFT")
    copy = item("copy", "UPDATE 1-Microsoft beats estimates", hours=2, ticker="MSFT")
    assert [n.id for n in relevant_unique([off, on, copy])] == ["on"]
