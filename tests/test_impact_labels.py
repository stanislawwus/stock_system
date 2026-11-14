"""gap_ar / drift_ar definitions (docs/02 "Labels")."""

from __future__ import annotations

from datetime import date

import pytest

from src.impact.labels import abnormal_returns

D0, D1, D2, D3 = date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3), date(2026, 6, 4)
SESSIONS = [D0, D1, D2, D3]
# (open, close)
STOCK = {D0: (100.0, 100.0), D1: (102.0, 103.0), D2: (104.0, 106.0), D3: (106.0, 107.0)}
MARKET = {D0: (400.0, 400.0), D1: (400.0, 401.0), D2: (402.0, 404.0), D3: (404.0, 405.0)}


def test_gap_and_drift_are_computed_against_the_market() -> None:
    r = abnormal_returns(D1, STOCK, MARKET, SESSIONS)
    assert r is not None and r.decision_day == D1
    # gap: close(D0)->open(D1): stock +2 %, market 0 %
    assert r.gap_ar == pytest.approx(0.02 - 0.0)
    # drift: open(D1)->close(D2): stock 106/102-1, market 404/400-1
    assert r.drift_ar == pytest.approx((106 / 102 - 1) - (404 / 400 - 1))


def test_edges_and_missing_prices_give_none() -> None:
    assert abnormal_returns(D0, STOCK, MARKET, SESSIONS) is None  # no D-1
    assert abnormal_returns(D3, STOCK, MARKET, SESSIONS) is None  # no D+1
    assert abnormal_returns(date(2026, 6, 6), STOCK, MARKET, SESSIONS) is None  # not a session
    gappy = {k: v for k, v in STOCK.items() if k != D2}
    assert abnormal_returns(D1, gappy, MARKET, SESSIONS) is None


def test_a_move_after_d_plus_one_does_not_change_the_label() -> None:
    later = {**STOCK, D3: (999.0, 999.0)}
    assert abnormal_returns(D1, later, MARKET, SESSIONS) == abnormal_returns(
        D1, STOCK, MARKET, SESSIONS
    )
