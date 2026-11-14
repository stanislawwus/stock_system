"""Relevance (does a news item talk about its tagged ticker?) and near-duplicate removal (D-36).

- Relevance: the ticker symbol or a company alias from `config/nlp.yaml` appears in the headline
  or summary. Finnhub tags an article with every company it touches (32 % of articles carry
  several tickers, docs/04), so most items under a ticker are about someone else.
- Dedup: MinHash over character shingles of the normalised headline. Items are processed in
  `available_at` order and an item is dropped when it is a near-duplicate of an earlier *kept*
  item of the same ticker within `window_hours`. The earliest copy survives, so whether an item
  is kept depends only on items published before it: running dedup on a whole split or inside
  `MarketView(as_of)` gives the same answer for every item visible at `as_of` (R2).

Pure functions on `NewsItem`s, no I/O apart from the cached config.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from typing import Any

import numpy as np
import yaml

from src.collectors.common import CONFIG_DIR
from src.nlp.clean import dedup_key, normalize_text
from src.schemas import NewsItem
from src.textutil import strip_html

_PRIME = np.uint64((1 << 31) - 1)
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})


@lru_cache(maxsize=1)
def config() -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / "nlp.yaml").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- relevance


def _clean(text: str) -> str:
    return normalize_text(strip_html(text)).translate(_APOSTROPHES)


@lru_cache(maxsize=512)
def _patterns(ticker: str) -> tuple[re.Pattern[str], ...]:
    cfg = config()["relevance"]
    t = re.escape(ticker)
    # marked symbol: $T, (T), NYSE: T, NASDAQ: T, NYSE:T
    pats = [re.compile(rf"(?:\${t}|\({t}\)|(?:NYSE|NASDAQ|Nasdaq)\s*:\s*{t})(?![A-Za-z])")]
    if len(ticker) >= int(cfg["min_bare_symbol_len"]) and ticker not in cfg["ambiguous_symbols"]:
        pats.append(re.compile(rf"(?<![A-Za-z0-9$]){t}(?![A-Za-z0-9])"))
    for alias in cfg["aliases"].get(ticker, []):
        a = re.escape(alias.translate(_APOSTROPHES))
        pats.append(re.compile(rf"(?<![A-Za-z0-9]){a}(?![A-Za-z0-9])", re.IGNORECASE))
    return tuple(pats)


@lru_cache(maxsize=512)
def _exclusions(ticker: str) -> re.Pattern[str] | None:
    phrases = config()["relevance"].get("exclude_phrases", {}).get(ticker, [])
    if not phrases:
        return None
    alts = "|".join(re.escape(p.translate(_APOSTROPHES)) for p in phrases)
    return re.compile(rf"(?<![A-Za-z0-9])(?:{alts})(?![A-Za-z0-9])", re.IGNORECASE)


def mentions(ticker: str, text: str) -> bool:
    """True when `text` names `ticker` by symbol or alias (see module docstring). Phrases in
    `relevance.exclude_phrases` (e.g. "J.P. Morgan Healthcare Conference") are blanked first."""
    clean = _clean(text)
    excl = _exclusions(ticker)
    if excl is not None:
        clean = excl.sub(" ", clean)
    return any(p.search(clean) for p in _patterns(ticker))


def is_relevant(item: NewsItem) -> bool:
    return mentions(item.ticker, f"{item.headline}\n{item.summary}")


# --------------------------------------------------------------------------- MinHash dedup


@dataclass(frozen=True)
class MinHasher:
    num_perm: int
    shingle_chars: int
    seed: int

    @classmethod
    def from_config(cls) -> MinHasher:
        c = config()["dedup"]
        return cls(int(c["num_perm"]), int(c["shingle_chars"]), int(c["seed"]))

    @property
    def _params(self) -> tuple[np.ndarray, np.ndarray]:
        return _perm_params(self.num_perm, self.seed)

    def shingles(self, text: str) -> set[str]:
        key = dedup_key(text)
        k = self.shingle_chars
        if len(key) <= k:
            return {key}
        return {key[i : i + k] for i in range(len(key) - k + 1)}

    def signature(self, text: str) -> np.ndarray:
        """Vector of `num_perm` minima of universal hashes (a*x + b) mod p over the shingles.
        p = 2^31 - 1 keeps a*x below 2^62, so uint64 arithmetic cannot overflow."""
        x = np.array(
            [
                int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")
                % int(_PRIME)
                for s in self.shingles(text)
            ],
            dtype=np.uint64,
        )
        a, b = self._params
        return ((a[:, None] * x[None, :] + b[:, None]) % _PRIME).min(axis=1)


@lru_cache(maxsize=8)
def _perm_params(num_perm: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    hi = int(_PRIME)
    a = rng.integers(1, hi, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, hi, size=num_perm, dtype=np.uint64)
    return a, b


def jaccard_estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))


@dataclass(frozen=True)
class DedupResult:
    kept: list[NewsItem]
    dropped: dict[str, str]  # dropped news id -> id of the kept item it duplicates


def dedup(
    items: Iterable[NewsItem],
    *,
    threshold: float | None = None,
    window_hours: float | None = None,
    hasher: MinHasher | None = None,
) -> DedupResult:
    """Drop near-duplicate headlines per ticker, keeping the earliest copy (see module doc)."""
    cfg = config()["dedup"]
    thr = float(cfg["threshold"] if threshold is None else threshold)
    window = timedelta(hours=float(cfg["window_hours"] if window_hours is None else window_hours))
    mh = hasher or MinHasher.from_config()
    ordered = sorted(items, key=lambda n: (n.available_at, n.id))
    kept: list[NewsItem] = []
    dropped: dict[str, str] = {}
    by_ticker: dict[str, list[tuple[NewsItem, np.ndarray]]] = {}
    for item in ordered:
        sig = mh.signature(item.headline)
        recent = [
            (k, s)
            for k, s in by_ticker.get(item.ticker, [])
            if item.available_at - k.available_at <= window
        ]
        by_ticker[item.ticker] = recent
        match = _best_match(sig, recent, thr)
        if match is not None:
            dropped[item.id] = match.id
            continue
        recent.append((item, sig))
        kept.append(item)
    return DedupResult(kept, dropped)


def _best_match(
    sig: np.ndarray, recent: Sequence[tuple[NewsItem, np.ndarray]], thr: float
) -> NewsItem | None:
    if not recent:
        return None
    sims = (np.stack([s for _, s in recent]) == sig).mean(axis=1)
    i = int(np.argmax(sims))
    return recent[i][0] if sims[i] >= thr else None


def relevant_unique(items: Iterable[NewsItem]) -> list[NewsItem]:
    """Relevant items with near-duplicates removed (relevance first: an off-topic copy must not
    knock out an on-topic original)."""
    return dedup([n for n in items if is_relevant(n)]).kept
