"""R3: no code path except `make test-final` reads test-period rows (D-33).

Three layers: (1) behaviour: MarketView refuses test-period views outside `final_run()`;
(2) static: only an explicit allowlist of files may mention the test period; (3) Makefile: only
`test-final` (and `collect-test`, which writes) names it.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from src import db, splits
from src.marketview import MarketView, cutoff_for
from src.splits import TestPeriodLockedError

ROOT = Path(__file__).resolve().parents[1]
FIRST_TEST_DAY = date(2026, 9, 1)
FIRST_CUTOFF = cutoff_for(FIRST_TEST_DAY)


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "t.duckdb"
    ts = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)  # a test-period row, before the 09:00 ET cutoff
    with db.writer_session(path) as con:
        con.execute("INSERT INTO news VALUES ('n', 'AAPL', ?, 'h', '', 'u', 'test', ?)", [ts, ts])
    return path


def test_first_test_cutoff_is_derived_from_the_frozen_splits() -> None:
    assert splits.first_test_cutoff() == FIRST_CUTOFF


def test_view_before_the_first_test_cutoff_is_allowed(db_path: Path) -> None:
    just_before = FIRST_CUTOFF - timedelta(seconds=1)
    assert [n.id for n in MarketView(just_before, db_path).news("AAPL")] == ["n"]


@pytest.mark.parametrize(
    "as_of",
    [
        FIRST_CUTOFF,
        FIRST_CUTOFF + timedelta(days=30),
        datetime(2026, 11, 2, 14, 0, tzinfo=UTC),
    ],
)
def test_view_in_the_test_period_is_refused(as_of: datetime, db_path: Path) -> None:
    with pytest.raises(TestPeriodLockedError):
        MarketView(as_of, db_path)
    with pytest.raises(TestPeriodLockedError):
        MarketView.for_trading_day(as_of.date(), db_path)


def test_final_run_opens_the_period_and_closes_it_again(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(splits, "FINAL_LOCK", tmp_path / "TEST_FINAL.lock")
    as_of = cutoff_for(FIRST_TEST_DAY + timedelta(days=1))
    with splits.final_run():
        assert MarketView(as_of, db_path).news("AAPL")
    with pytest.raises(TestPeriodLockedError):
        MarketView(as_of, db_path)


def test_final_run_refuses_once_the_lock_file_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "TEST_FINAL.lock"
    lock.write_text("run_at=x\n", encoding="utf-8")
    monkeypatch.setattr(splits, "FINAL_LOCK", lock)
    with pytest.raises(TestPeriodLockedError), splits.final_run():
        pass


# Files that may name the test period, and why. Adding one needs a reason here (R3).
ALLOWED = {
    "src/splits.py": "the lock itself",
    "src/collectors/common.py": "period loader; the collector writes test rows only on request",
    "src/collectors/quality.py": "checks that a dev collection holds no test rows",
    "src/collectors/report.py": "cuts the coverage report before the test cutoff",
    "src/collectors/run.py": "collector CLI: --periods test is make collect-test",
    "src/eda/data.py": "offline analysis cut at the first test cutoff (D-33)",
    "src/schemas.py": "request/response Literals",
    "evals/eval_collectors.py": "plants a test-period row to prove the checks catch it",
}
PATTERN = re.compile(
    r"test_period\(|first_test_cutoff|final_run\(|\[.test.\]|--final|periods test"
    r"|\"test\"\s*[,\]]|load_periods\(\[[^\]]*test"
)


def _sources() -> list[Path]:
    return [p for d in ("src", "evals") for p in sorted((ROOT / d).rglob("*.py"))]


def _rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def test_only_allowlisted_files_mention_the_test_period() -> None:
    offenders = [
        _rel(p)
        for p in _sources()
        if PATTERN.search(p.read_text(encoding="utf-8")) and _rel(p) not in ALLOWED
    ]
    assert not offenders, f"R3: not allowed to reference the test period: {offenders}"


def test_the_allowlist_has_no_stale_entries() -> None:
    stale = [f for f in ALLOWED if not PATTERN.search((ROOT / f).read_text(encoding="utf-8"))]
    assert not stale, f"remove from ALLOWED, they no longer mention the test period: {stale}"


def test_final_run_is_opened_only_where_the_lock_is_defined_for_now() -> None:
    users = [_rel(p) for p in _sources() if re.search(r"\bfinal_run\(", p.read_text("utf-8"))]
    assert users == ["src/splits.py"], (
        "final_run() may be opened only from the --final entry points of make test-final "
        f"(src.strategies.arena, src.sim.run); add them here when they exist: {users}"
    )


def test_makefile_names_the_test_period_only_in_test_final_and_collect_test() -> None:
    target = None
    named: dict[str, list[str]] = {}
    for line in (ROOT / "Makefile").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([a-z-]+):", line)
        if m:
            target = m.group(1)
        if re.search(r"--final|--periods test|--split test|--splits test", line):
            named.setdefault(target or "?", []).append(line.strip())
    assert set(named) <= {"test-final", "collect-test"}, named
