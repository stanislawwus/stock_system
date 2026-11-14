"""Fact extraction from EX-99.1 earnings releases: {metric, value, unit, period} + grounding (D-39).

- `extract_regex`: deterministic baseline and the LLM fallback (`facts: xbrl_or_regex` in
  config/llm.yaml). Per metric the first GAAP-looking match wins: a match preceded within
  `regex_context_chars` by "adjusted", "non-GAAP", "organic" or "excluding" is skipped, because
  XBRL carries GAAP values.
- `extract_llm`: the same output schema through `src/llm.py` (batched, cached, budgeted); the
  release text is wrapped as untrusted data (R6). The model returns the number as written plus a
  scale word; code does the multiplication, the model never does arithmetic.
- `is_grounded`: the quote must occur in the source (whitespace-normalised) and a number in the
  quote, scaled by the word after it, must equal the value. Ungrounded facts are dropped by
  `grounded_only`, never repaired.

Text in, pydantic out; no I/O apart from the cached config.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from src.collectors.common import CONFIG_DIR
from src.textutil import strip_html

Metric = Literal["revenue", "eps_diluted", "net_income"]
Scale = Literal["one", "thousand", "million", "billion"]
SCALES: dict[str, float] = {"one": 1.0, "thousand": 1e3, "million": 1e6, "billion": 1e9}
_SCALE_WORDS = {"thousand": "thousand", "million": "million", "billion": "billion", "bn": "billion"}
_NUM = r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
_MONEY = rf"\$\s?{_NUM}\s*(billion|million|thousand|bn)\b"
_VERB = (
    r"(?:of|was|were|totaled|totaling|reached|came in at|of approximately|"
    r"(?:increased|decreased|grew|rose|fell|declined)[^.$]{0,40}?to)"
)
_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "revenue": [
        re.compile(
            rf"(?:total\s+)?(?:net\s+)?(?:revenues?|net\s+sales|sales)\s+{_VERB}\s+{_MONEY}", re.I
        ),
        re.compile(
            rf"{_MONEY}\s+(?:in|of)\s+(?:total\s+)?(?:net\s+)?(?:revenues?|net\s+sales)", re.I
        ),
    ],
    "net_income": [
        re.compile(
            rf"net\s+(?:income|earnings)(?:\s+attributable\s+to\s+[A-Za-z .,&]{{1,40}}?)?"
            rf"\s+{_VERB}\s+{_MONEY}",
            re.I,
        ),
    ],
    "eps_diluted": [
        re.compile(
            r"diluted\s+(?:earnings|net\s+income|net\s+earnings|EPS)\s*(?:per\s+(?:common\s+)?share)?"
            r"[^$.]{0,40}?\$\s?(\d+\.\d{2})",
            re.I,
        ),
        re.compile(r"\$\s?(\d+\.\d{2})\s+per\s+diluted\s+share", re.I),
    ],
}
_NON_GAAP = re.compile(r"adjusted|non-gaap|organic|excluding|core\b", re.I)
_PERIOD = re.compile(
    r"(first|second|third|fourth)[- ]quarter(?:\s+(?:of\s+)?(?:fiscal\s+(?:year\s+)?)?(\d{4}))?"
    r"|(?:fiscal\s+)?(?:(\d{4})\s+)?(first|second|third|fourth)\s+quarter",
    re.I,
)
_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4}
_QUOTE_NUM = re.compile(rf"{_NUM}\s*(billion|million|thousand|bn)?", re.I)


class ExtractedFact(BaseModel):
    """One extracted number; the LLM structured-output schema (R13)."""

    metric: Metric
    number: float = Field(description="the number exactly as written in the quote, no scaling")
    scale: Scale = Field(description="scale word written after the number; 'one' if none")
    unit: Literal["USD", "USD/share"]
    period: str = Field(max_length=40, description="fiscal period named in the text, e.g. Q3 2025")
    quote: str = Field(max_length=300, description="verbatim sentence fragment holding the number")

    @property
    def value(self) -> float:
        return self.number * SCALES[self.scale]


class FilingFacts(BaseModel):
    facts: list[ExtractedFact] = Field(default_factory=list, max_length=12)


@lru_cache(maxsize=1)
def config() -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / "extraction.yaml").read_text(encoding="utf-8"))["facts"]


def _ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def snippet(text: str) -> str:
    """Press-release text -> the model/regex input: no HTML, normalised spaces, bounded length."""
    return _ws(strip_html(text))[: int(config()["snippet_chars"])]


def find_period(text: str) -> str:
    m = _PERIOD.search(text[:1500])
    if not m:
        return ""
    if m.group(1):
        q, year = _ORD[m.group(1).lower()], m.group(2) or ""
    else:
        q, year = _ORD[m.group(4).lower()], m.group(3) or ""
    return f"Q{q} {year}".strip()


def extract_regex(text: str) -> FilingFacts:
    src = snippet(text)
    ctx = int(config()["regex_context_chars"])
    period = find_period(src)
    out: list[ExtractedFact] = []
    for metric, pats in _PATTERNS.items():
        found = None
        for pat in pats:
            for m in pat.finditer(src):
                if _NON_GAAP.search(src[max(0, m.start() - ctx) : m.end()]):
                    continue
                found = m
                break
            if found:
                break
        if found is None:
            continue
        groups = [g for g in found.groups() if g]
        number = float(groups[0].replace(",", ""))
        scale = _SCALE_WORDS[groups[1].lower()] if len(groups) > 1 else "one"
        out.append(
            ExtractedFact(
                metric=metric,  # type: ignore[arg-type]
                number=number,
                scale=scale,  # type: ignore[arg-type]
                unit="USD/share" if metric == "eps_diluted" else "USD",
                period=period,
                quote=found.group(0)[:300],
            )
        )
    return FilingFacts(facts=out)


def is_grounded(source: str, fact: ExtractedFact) -> bool:
    """The quote occurs in the source and one of its numbers (with its scale word) is the value."""
    quote = _ws(fact.quote)
    if not quote or quote not in _ws(strip_html(source)):
        return False
    for m in _QUOTE_NUM.finditer(quote):
        n = float(m.group(1).replace(",", ""))
        word = (m.group(2) or "").lower()
        scaled = n * SCALES[_SCALE_WORDS[word]] if word else n
        if abs(scaled - fact.value) <= 1e-9 * max(1.0, abs(fact.value)):
            return True
    return False


def grounded_only(source: str, facts: FilingFacts) -> FilingFacts:
    return FilingFacts(facts=[f for f in facts.facts if is_grounded(source, f)])


PROMPT = (
    "Each item is the start of a US company's quarterly earnings press release (SEC 8-K exhibit "
    "99.1). Extract the GAAP figures for the quarter the release reports: total revenue (or net "
    "sales), net income, and diluted earnings per share. Skip adjusted / non-GAAP / segment / "
    "year-to-date / guidance figures. For each figure give: metric (revenue | net_income | "
    "eps_diluted); number exactly as written; scale = the word written right after the number "
    "(thousand | million | billion) or 'one'; unit USD or USD/share; period as named in the text; "
    "quote = a verbatim fragment of the text (max 300 chars) containing the number. Omit a figure "
    "that is not stated. Return {'facts': [...]} per item."
)


def extract_llm(texts: Sequence[str]):  # -> LLMResult[FilingFacts]
    """Batched LLM extraction with the regex baseline as the deterministic fallback."""
    from src import llm

    items = [{"release": snippet(t)} for t in texts]
    return llm.call_json(
        "facts",
        PROMPT,
        items,
        FilingFacts,
        fallback=lambda batch: [extract_regex(b["release"]) for b in batch],
    )
