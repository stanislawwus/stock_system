"""Text cleaning shared by data-quality checks, sampling and the NLP components (docs/04).

Pure functions, no I/O. Thresholds come from `config/data_quality.yaml` (R14).
News and filings are untrusted text (R6): nothing here executes or interprets it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import yaml

from src.collectors.common import CONFIG_DIR

ENGLISH_WORDS = frozenset(
    "the and of to in for on with is are at as by from after before over amid says new".split()
)
FOREIGN_WORDS = frozenset(
    "de la el los las del que por para con una der die das und nicht mit für le les des du et "
    "pour dans sur il di che della per een het van och på med för ve için bir".split()
)

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


@lru_cache(maxsize=1)
def config() -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / "data_quality.yaml").read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    """NFKC (folds no-break spaces and full-width forms), collapse whitespace, strip."""
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def dedup_key(headline: str) -> str:
    """Lower-cased headline without punctuation: two syndicated copies share a key."""
    return _SPACE_RE.sub(" ", _PUNCT_RE.sub(" ", normalize_text(headline).casefold())).strip()


def latin_share(text: str) -> float:
    """Share of letters written in Latin script (accented letters count); 1.0 without letters."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 1.0
    latin = sum(1 for ch in letters if unicodedata.name(ch, "").startswith("LATIN"))
    return latin / len(letters)


def is_probably_english(text: str) -> bool:
    """Heuristic, not a language identifier: non-Latin script or foreign function words dominate."""
    cfg = config()["language"]
    if latin_share(text) < float(cfg["min_latin_share"]):
        return False
    words = [w.casefold() for w in _WORD_RE.findall(text)]
    foreign = len(set(words) & FOREIGN_WORDS)
    english = len(set(words) & ENGLISH_WORDS)
    return not (foreign >= int(cfg["foreign_min_hits"]) and foreign > english)


@dataclass(frozen=True)
class HeadlineCheck:
    ok: bool
    problems: tuple[str, ...]


def check_headline(headline: str) -> HeadlineCheck:
    """Reasons a headline is unfit for sentiment/event labelling: too_short, too_long, listicle,
    non_english. An empty tuple means usable."""
    cfg = config()["headline"]
    text = normalize_text(headline)
    problems = []
    if len(text) < int(cfg["min_chars"]):
        problems.append("too_short")
    if len(text) > int(cfg["max_chars"]):
        problems.append("too_long")
    if re.match(cfg["listicle_pattern"], text):
        problems.append("listicle")
    if not is_probably_english(text):
        problems.append("non_english")
    return HeadlineCheck(not problems, tuple(problems))


_TAG_LINE_RE = re.compile(r"^\[[A-Za-z0-9.\-]+\][^\n]*\n?")
_ITEM_RE = re.compile(r"^\s*item\s+\d{1,2}\.\d{2}\b", re.IGNORECASE)
_SECTION_RE = re.compile(r"^\s*section\s+\d{1,2}\s*[-–—:.]", re.IGNORECASE)
_INLINE_ITEM_RE = re.compile(r"\bitems?\s+\d{1,2}\.\d{2}(\([a-z0-9]\))*", re.IGNORECASE)


def strip_8k_item_headers(text: str) -> str:
    """Remove what names the event type before an 8-K text reaches a classifier.

    The 8-K item numbers are the gold labels (D-10), so a text that still says "Item 5.02
    Departure of Directors ..." would make the event eval circular. Removed: the collector's
    `[EX-99.1] ...` tag line, `Item x.xx` headings with their wrapped title (up to
    `filing_text.header_max_lines` lines, ending at a blank line or a full stop),
    `Section n - ...` headings, and inline mentions such as "under Item 2.02", which become
    "under the item". The body text is left untouched.
    """
    limit = int(config()["filing_text"]["header_max_lines"])
    lines = _TAG_LINE_RE.sub("", text, count=1).split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _SECTION_RE.match(line):
            i += 1
        elif _ITEM_RE.match(line):
            taken = 1
            while not lines[i].rstrip().endswith(".") and taken < limit:
                nxt = i + 1
                if nxt >= len(lines) or not lines[nxt].strip():
                    break
                i, taken = nxt, taken + 1
            i += 1
        else:
            out.append(_INLINE_ITEM_RE.sub("the item", line))
            i += 1
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
