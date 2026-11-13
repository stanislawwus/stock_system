"""Plain-text helpers shared by collectors and the LLM gateway (R6: strip HTML from web text)."""

from __future__ import annotations

import html
import re

_SCRIPT_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_BLOCK_RE = re.compile(r"<\s*(br|/p|/div|/tr|/li|/h\d)\b[^>]*>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACES_RE = re.compile(r"[ \t\r\f\v ]+")
_NEWLINES_RE = re.compile(r"\n\s*\n+")


def strip_html(text: str, *, keep_lines: bool = False) -> str:
    """Remove scripts, styles and tags, unescape entities, normalise whitespace.

    With `keep_lines`, block-level tags become newlines (readable filings text); otherwise the
    result is a single line (compact prompt input).
    """
    text = _SCRIPT_RE.sub(" ", text)
    if keep_lines:
        text = _BLOCK_RE.sub("\n", text)
    text = html.unescape(_TAG_RE.sub(" ", text))
    if not keep_lines:
        return " ".join(text.split())
    lines = [_SPACES_RE.sub(" ", ln).strip() for ln in text.split("\n")]
    return _NEWLINES_RE.sub("\n\n", "\n".join(lines)).strip()
