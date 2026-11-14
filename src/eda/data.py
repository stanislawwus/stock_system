"""Read access for offline analysis (data quality, EDA, label sheets). Development data only.

Not a decision path (D-33): strategies, NLP and impact code still read through `MarketView`.
Every frame built here is cut at the first test cutoff, so the test period stays unread even if
`make collect-test` has been run (AGENTS.md R3).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from src.db import connect
from src.marketview.clock import cutoff_for
from src.splits import first_test_cutoff


def open_dev(db_path: Any = None) -> duckdb.DuckDBPyConnection:
    """Read-only connection whose timestamps come back in UTC, whatever the machine's zone."""
    con = connect(db_path)
    con.execute("SET TimeZone = 'UTC'")
    return con


def dev_cut() -> datetime:
    return first_test_cutoff()


def frame(
    con: duckdb.DuckDBPyConnection, sql: str, params: list[Any] | None = None
) -> pd.DataFrame:
    """Run a query that must already contain its own `available_at < ?` bound (`dev_cut()`)."""
    return con.execute(sql, params or []).fetch_df()


def news_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return frame(
        con,
        "SELECT id, ticker, published_at, headline, summary, url, available_at "
        "FROM news WHERE available_at < ?",
        [dev_cut()],
    )


def sessions(con: duckdb.DuckDBPyConnection, benchmark: str = "SPY") -> list[date]:
    """Exchange sessions seen so far: the benchmark's bar dates (D-29), development only."""
    rows = con.execute(
        "SELECT bar_date FROM price_bars WHERE ticker = ? AND available_at < ? ORDER BY bar_date",
        [benchmark, dev_cut()],
    ).fetchall()
    return [r[0] for r in rows]


def decision_days(available_at: pd.Series, session_days: list[date]) -> pd.Series:
    """Vectorised `clock.decision_day`: first session whose 09:00 ET cutoff is >= the timestamp.

    NaT where no listed session qualifies. Same rule, one searchsorted instead of a loop.
    """
    cutoffs = pd.DatetimeIndex([cutoff_for(d) for d in session_days]).tz_convert("UTC")
    ts = pd.DatetimeIndex(pd.to_datetime(available_at, utc=True))
    idx = np.searchsorted(
        cutoffs.as_unit("ns").tz_convert(None).to_numpy(),
        ts.as_unit("ns").tz_convert(None).to_numpy(),
        side="left",
    )
    days = np.array(session_days + [None], dtype=object)
    return pd.Series(days[idx], index=available_at.index)
