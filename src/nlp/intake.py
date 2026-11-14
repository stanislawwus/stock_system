"""Intake: company name -> ticker and watchlist phrase -> tickers (docs/01 Must; D-45).

Deterministic table resolver, the fallback path of the intake agent (docs/01: "LLM with table
fallback"; the LLM path is not built). The table is the universe tickers (config/universe.yaml)
plus the company aliases already used for news relevance (config/nlp.yaml `relevance.aliases`);
SEC company titles are *not* used, because the intake eval derives its gold set from them.

- `resolve_name(name)`: one company name (or a typed ticker / $ticker) -> ticker or None. An
  alias must match whole words at the start of the name, or the rest of the name must be only
  legal-form words (Inc, Corp, Group, ...). Longest alias wins.
- `extract_tickers(phrase)`: every company named in a free-text request, in order of first
  mention: $TICKER, an upper-case universe ticker (symbols that are also words, e.g. ALL, NOW,
  only with $), and aliases anywhere on word boundaries (longest first, no overlaps).
User text is data, never instructions (R6): nothing here calls a model or executes anything.
"""

from __future__ import annotations

import re
from functools import lru_cache

from src.collectors.common import load_universe, load_yaml

LEGAL_FORMS = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies", "plc", "ltd",
    "limited", "holdings", "holding", "group", "the", "and", "de", "class", "a", "b", "n", "v",
    "sa", "ag", "nv", "lp", "llc", "new",
}  # fmt: skip
TOKEN = re.compile(r"[a-z0-9]+")


def normalise(text: str) -> list[str]:
    """Lower-case word tokens; '&' reads as 'and', punctuation splits ("J.P." -> j, p)."""
    return TOKEN.findall(text.lower().replace("&", " and "))


@lru_cache(maxsize=1)
def tables() -> tuple[frozenset[str], frozenset[str], tuple[tuple[tuple[str, ...], str], ...]]:
    """(universe tickers, symbols that need a $, aliases as token tuples longest first)."""
    tickers = frozenset(c.ticker for c in load_universe()[0])
    rel = load_yaml("nlp.yaml")["relevance"]
    ambiguous = frozenset(rel["ambiguous_symbols"])
    aliases = {(tuple(normalise(a)), t) for t, names in rel["aliases"].items() for a in names}
    ordered = tuple(sorted((a for a in aliases if a[0]), key=lambda x: (-len(x[0]), x[1], x[0])))
    return tickers, ambiguous, ordered


def _typed_ticker(token: str, tickers: frozenset[str], ambiguous: frozenset[str]) -> str | None:
    if token.startswith("$") and token[1:].upper() in tickers:
        return token[1:].upper()
    if token.isupper() and token in tickers and token not in ambiguous:
        return token
    return None


def resolve_name(name: str) -> str | None:
    tickers, ambiguous, aliases = tables()
    stripped = name.strip()
    typed = _typed_ticker(stripped, tickers, ambiguous)
    if typed or (stripped.startswith("$") and stripped[1:].upper() in tickers):
        return typed or stripped[1:].upper()
    toks = normalise(stripped)
    for alias, ticker in aliases:
        n = len(alias)
        for i in range(len(toks) - n + 1):
            if tuple(toks[i : i + n]) != alias:
                continue
            rest = toks[:i] + toks[i + n :]
            if i == 0 or all(t in LEGAL_FORMS for t in rest):
                return ticker
    return None


def extract_tickers(phrase: str) -> list[str]:
    tickers, ambiguous, aliases = tables()
    # A request is not news prose: a bare upper-case universe ticker (MA, GE, C) is meant as a
    # ticker, unless the whole request is written in capitals; then only $TICKER counts.
    letters = [c for c in phrase if c.isalpha()]
    shouting = bool(letters) and all(c.isupper() for c in letters)
    need_dollar = tickers if shouting else frozenset()
    found: list[tuple[int, str]] = []  # (position in the phrase, ticker)
    for m in re.finditer(r"\$?[A-Za-z][A-Za-z.\-]*", phrase):
        t = _typed_ticker(m.group(0).rstrip(".,"), tickers, need_dollar)
        if t:
            found.append((m.start(), t))
    text = phrase.replace("&", " and ")  # same reading as `normalise`; positions only order
    words = list(TOKEN.finditer(text.lower()))
    toks = [w.group(0) for w in words]
    used = [False] * len(toks)
    for alias, ticker in aliases:
        n = len(alias)
        for i in range(len(toks) - n + 1):
            if tuple(toks[i : i + n]) == alias and not any(used[i : i + n]):
                used[i : i + n] = [True] * n
                found.append((words[i].start(), ticker))
    out: list[str] = []
    for _, t in sorted(found):
        if t not in out:
            out.append(t)
    return out
