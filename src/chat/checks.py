"""Code checks for chat answers (D-46): no LLM judge, everything is string and set logic.

- `numbers_ok`: every number in the answer, outside untrusted quoted spans, appears in the tool
  outputs (at the precision the composer prints);
- `citation_markers_ok`: [n] markers and the citation list match one-to-one;
- `wording_ok`: no causal verbs outside quoted data; the disclaimer is present;
- `rephrase_ok`: a rephrased answer keeps the template's numbers, markers and quoted spans.
Citation validity against the database lives in evals/eval_chat.py (it needs a read connection).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, date, datetime
from typing import Any

from src.llm import DATA_CLOSE, DATA_OPEN
from src.schemas import DISCLAIMER, Citation

UNTRUSTED = re.compile(re.escape(DATA_OPEN) + r".*?" + re.escape(DATA_CLOSE), re.DOTALL)
MARKER = re.compile(r"\[(\d+)\]")
FORM = re.compile(r"\b(?:8-K|10-Q|10-K)\b")  # form names, not numbers
NUMBER = re.compile(r"\d+(?:[.:\-]\d+)*")
CAUSAL = re.compile(
    r"\b(caused|causes|causing|drove|driven by|drives|explains?|explained|because of|due to|"
    r"led to|triggered|resulted in)\b",
    re.IGNORECASE,
)


def outside_quotes(text: str) -> str:
    """The answer with untrusted quoted spans blanked out: what the system itself claims."""
    return UNTRUSTED.sub(" ", text)


def answer_numbers(text: str) -> list[str]:
    body = FORM.sub(" ", MARKER.sub(" ", outside_quotes(text)))
    return NUMBER.findall(body)


def _fmt_number(v: float) -> set[str]:
    return {f"{v:.2f}", f"{abs(v):.2f}", f"{abs(v) * 100:.2f}", f"{v:.0f}", f"{abs(v):.0f}"}


def _walk(obj: Any, acc: set[str]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, int | float):
        acc |= _fmt_number(float(obj))
        if isinstance(obj, int):
            acc.add(str(obj))
    elif isinstance(obj, datetime):
        _walk(obj.isoformat(), acc)
    elif isinstance(obj, date):
        acc.add(obj.isoformat())
    elif isinstance(obj, str):
        acc.add(obj)
        acc.update(NUMBER.findall(obj))
        try:  # ISO timestamps from model_dump(mode="json"): also the printed minute and day
            ts = datetime.fromisoformat(obj.replace("Z", "+00:00"))
        except ValueError:
            return
        if ts.tzinfo is not None:
            u = ts.astimezone(UTC)
            acc.update({u.date().isoformat(), u.strftime("%H:%M")})
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk(v, acc)
    elif isinstance(obj, Iterable):
        for v in obj:
            _walk(v, acc)


def allowed_numbers(tool_outputs: dict[str, Any], question: str = "") -> set[str]:
    """Every string a number in the answer may legitimately take, from tool outputs. Numbers in
    the question (an asked date that was not a session) and in the disclaimer are allowed too."""
    acc: set[str] = set()
    _walk(tool_outputs, acc)
    acc.update(NUMBER.findall(question))
    acc.update(NUMBER.findall(DISCLAIMER))
    acc.update({"0", "09:00"})  # "score <= 0 means cash", "09:00 ET cutoff" (template constants)
    return acc


def unsupported_numbers(answer: str, tool_outputs: dict[str, Any], question: str = "") -> list[str]:
    ok = allowed_numbers(tool_outputs, question)
    return [n for n in answer_numbers(answer) if n not in ok]


def numbers_ok(answer: str, tool_outputs: dict[str, Any], question: str = "") -> bool:
    return not unsupported_numbers(answer, tool_outputs, question)


def citation_markers_ok(answer: str, citations: list[Citation]) -> bool:
    used = {int(m) for m in MARKER.findall(outside_quotes(answer))}
    listed = [c.n for c in citations]
    return len(listed) == len(set(listed)) and used == set(listed)


def wording_ok(answer: str) -> bool:
    return DISCLAIMER in answer and not CAUSAL.search(outside_quotes(answer))


def rephrase_ok(rephrased: str, template: str) -> bool:
    """Accept a rephrase only if it keeps exactly the template's numbers, [n] markers and quoted
    spans, adds no causal wording, and opens no new untrusted span."""
    return (
        Counter(answer_numbers(rephrased)) == Counter(answer_numbers(template))
        and sorted(MARKER.findall(outside_quotes(rephrased)))
        == sorted(MARKER.findall(outside_quotes(template)))
        and UNTRUSTED.findall(rephrased) == UNTRUSTED.findall(template)
        and rephrased.count(DATA_OPEN) == template.count(DATA_OPEN)
        and not CAUSAL.search(outside_quotes(rephrased))
    )
