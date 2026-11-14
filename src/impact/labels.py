"""Abnormal-return labels (docs/02_architecture.md "Labels", D-21, D-28). Pure functions.

For an event whose `available_at` falls before the 09:00 ET cutoff of trading day D (and after the
previous cutoff), with SPY as the market:

    gap_ar   = r(close(D-1) -> open(D))  - r_SPY(close(D-1) -> open(D))   priced in before our entry
    drift_ar = r(open(D)    -> close(D+1)) - r_SPY(open(D) -> close(D+1)) the tradable target

`gap_ar` needs the open of D, so it is a diagnostic: strategies never see it. These functions take
prices as arguments and read nothing, so the impact package never reaches for execution data;
callers (the EDA, the labeller job) fetch the series.
Labels are forward-looking by construction and live only in `impact_labels` (D-28).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

# (open, close) of one trading day
Bar = tuple[float, float]
OPEN, CLOSE = 0, 1


@dataclass(frozen=True)
class AbnormalReturns:
    decision_day: date
    gap_ar: float
    drift_ar: float


def _ret(a: float, b: float) -> float:
    return b / a - 1.0


def abnormal_returns(
    day: date,
    stock: Mapping[date, Bar],
    market: Mapping[date, Bar],
    sessions: Sequence[date],
) -> AbnormalReturns | None:
    """Labels for decision day `day`, or None when D-1, D or D+1 is missing for the stock, the
    market or the calendar (edges of the sample, halted days)."""
    if day not in sessions:
        return None
    i = sessions.index(day)
    if i == 0 or i + 1 >= len(sessions):
        return None
    prev, nxt = sessions[i - 1], sessions[i + 1]
    try:
        gap = _ret(stock[prev][CLOSE], stock[day][OPEN]) - _ret(
            market[prev][CLOSE], market[day][OPEN]
        )
        drift = _ret(stock[day][OPEN], stock[nxt][CLOSE]) - _ret(
            market[day][OPEN], market[nxt][CLOSE]
        )
    except KeyError:
        return None
    return AbnormalReturns(day, gap, drift)
