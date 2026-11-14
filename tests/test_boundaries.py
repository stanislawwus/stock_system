"""Static architecture boundaries (AGENTS.md R2, R7, R13). All pass now and must stay green."""

from __future__ import annotations

import ast
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


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
            mods |= {f"{node.module}.{a.name}" for a in node.names}  # `from src import db`
    return mods


def test_ui_package_never_imports_duckdb_or_the_database_layer() -> None:
    """AST check, so a comment or string cannot fool it: the UI reads through httpx only."""
    forbidden = ("duckdb", "src.db")
    bad = {}
    for p in _py(SRC / "ui"):
        hit = sorted(
            m
            for m in _imported_modules(p)
            if any(m == f or m.startswith(f + ".") for f in forbidden)
        )
        if hit:
            bad[str(p.relative_to(ROOT))] = hit
    assert not bad, f"UI imports the database layer: {bad}"


def test_ui_imports_from_src_are_only_schemas_and_itself() -> None:
    """The UI shares pydantic contracts with the API and nothing else of the backend."""
    allowed = ("src.schemas", "src.ui")
    bad = {}
    for p in _py(SRC / "ui"):
        extra = sorted(
            m for m in _imported_modules(p)
            if m.startswith("src.") and not any(m == a or m.startswith(a + ".") for a in allowed)
        )  # fmt: skip
        if extra:
            bad[str(p.relative_to(ROOT))] = extra
    assert not bad, f"UI imports backend modules: {bad}"


def test_servers_bind_to_loopback_only() -> None:
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "0.0.0.0" not in text
    assert "HOST ?= 127.0.0.1" in text
    assert text.count("--host $(HOST)") >= 2 and text.count("--server.address $(HOST)") >= 2
    ui_src = "\n".join(p.read_text(encoding="utf-8") for p in _py(SRC / "ui"))
    assert "0.0.0.0" not in ui_src
