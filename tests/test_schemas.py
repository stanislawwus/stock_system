"""Contracts in src/schemas.py: every record has available_at, bounds and invariants hold."""

from __future__ import annotations

import inspect
from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from src import schemas
from src.schemas import Event, Fact, Record, Signal, Span

AS_OF = datetime(2026, 9, 1, 13, 0, tzinfo=UTC)


def _record_models() -> list[type[Record]]:
    return [
        obj
        for _, obj in inspect.getmembers(schemas, inspect.isclass)
        if issubclass(obj, Record) and obj is not Record
    ]


def test_every_record_model_has_available_at() -> None:
    models = _record_models()
    assert {m.__name__ for m in models} >= {
        "PriceBar",
        "NewsItem",
        "Filing",
        "Fact",
        "Sentiment",
        "Event",
        "Signal",
        "Trade",
        "BacktestResult",
    }
    for m in models:
        assert "available_at" in m.model_fields, m.__name__


def test_available_at_must_be_timezone_aware() -> None:
    with pytest.raises(ValidationError):
        Event(
            available_at=datetime(2026, 9, 1, 13, 0),  # naive
            source_ref="n1",
            ticker="AAPL",
            labels=["2.02"],
            confidence=0.9,
            method="llm",
        )


def test_signal_defaults_available_at_to_as_of_and_bounds_score() -> None:
    s = Signal(ticker="AAPL", as_of=AS_OF, strategy="momentum", score=0.4, reasons=["ma50>ma200"])
    assert s.available_at == s.as_of
    with pytest.raises(ValidationError):
        Signal(ticker="AAPL", as_of=AS_OF, strategy="momentum", score=1.5)
    with pytest.raises(ValidationError):
        Signal(ticker="AAPL", as_of=AS_OF, strategy="m", score=0.1, calibrated_p=1.2)
    with pytest.raises(ValidationError):
        Signal(
            ticker="AAPL",
            as_of=AS_OF,
            strategy="m",
            score=0.1,
            available_at=datetime(2026, 9, 2, 13, 0, tzinfo=UTC),
        )


def test_event_is_multi_label_and_none_is_exclusive() -> None:
    e = Event(
        available_at=AS_OF,
        source_ref="acc-1",
        ticker="AAPL",
        labels=["2.02", "9.01"],
        confidence=0.8,
        method="index",
    )
    assert e.labels == ["2.02", "9.01"]
    with pytest.raises(ValidationError):
        Event(
            available_at=AS_OF,
            source_ref="acc-1",
            ticker="AAPL",
            labels=["none", "2.02"],
            confidence=0.8,
            method="index",
        )
    with pytest.raises(ValidationError):
        Event(
            available_at=AS_OF,
            source_ref="acc-1",
            ticker="AAPL",
            labels=[],
            confidence=0.8,
            method="index",
        )


def test_extracted_fact_requires_grounding_span() -> None:
    base = dict(
        available_at=AS_OF,
        ticker="AAPL",
        name="revenue",
        value=94.9e9,
        unit="USD",
        period_end=date(2026, 6, 27),
        fiscal_period="Q3-2026",
        source_ref="acc-1",
    )
    Fact(**base, method="xbrl")  # gold facts carry no span
    with pytest.raises(ValidationError):
        Fact(**base, method="llm")
    Fact(**base, method="llm", grounding=Span(source_ref="acc-1", start=10, end=42, quote="..."))
    with pytest.raises(ValidationError):
        Span(source_ref="acc-1", start=10, end=5, quote="x")


def test_records_are_frozen_and_reject_unknown_fields() -> None:
    s = Signal(ticker="AAPL", as_of=AS_OF, strategy="m", score=0.0)
    with pytest.raises(ValidationError):
        s.score = 1.0  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Signal(ticker="AAPL", as_of=AS_OF, strategy="m", score=0.0, extra_field=1)  # type: ignore[call-arg]
