"""Static architecture boundaries (AGENTS.md R2, R7, R13). All pass now and must stay green."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _py(under: Path) -> list[Path]:
    return sorted(p for p in under.rglob("*.py"))


def _offenders(files: list[Path], pattern: str) -> list[str]:
    rx = re.compile(pattern)
    return [str(p.relative_to(ROOT)) for p in files if rx.search(p.read_text(encoding="utf-8"))]


def test_ui_only_talks_to_the_api() -> None:
    bad = _offenders(_py(SRC / "ui"), r"\bduckdb\b|src\.db\b|src\.marketview|src\.llm\b")
    assert not bad, f"UI must use httpx only: {bad}"


def test_duckdb_is_opened_only_in_db_py_and_collectors() -> None:
    allowed = {"src/db.py"}
    files = [p for p in _py(SRC) if not str(p.relative_to(ROOT)).startswith("src/collectors")]
    found = _offenders(files, r"\bduckdb\.connect\(")
    bad = [f for f in found if f.replace("\\", "/") not in allowed]
    assert not bad, f"open DuckDB only via src.db.connect: {bad}"


def test_routers_contain_no_logic() -> None:
    bad = _offenders(
        _py(SRC / "api" / "routers"), r"\bduckdb\b|MarketView|call_json|pandas|numpy|src\.llm"
    )
    assert not bad, f"routers must only call services: {bad}"


def test_decision_modules_never_read_the_open_of_day_d() -> None:
    files = [p for d in ("strategies", "nlp", "impact") for p in _py(SRC / d)]
    bad = _offenders(files, r"open_price\(|execution_price|\.open\b\s*\]")
    assert not bad, f"only src/sim may read open(D): {bad}"
