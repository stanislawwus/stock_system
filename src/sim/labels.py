"""Forward-looking labels per (ticker, session) to fit and score the impact model (D-28, D-40).

`drift_ar` of D needs the open of D and the close of D+1, so it is built here, in `src/sim`, the
only package allowed to read execution prices (R2). The labels are used as training targets and
eval gold only; no strategy or feature ever receives them. D+1 must be inside the loaded range,
so the last session of a split has no label (no label crosses a split boundary).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.impact.labels import abnormal_returns
from src.sim.execution import ExecutionPrices


def _series(prices: ExecutionPrices, ticker: str) -> dict[date, tuple[float, float]]:
    out = {}
    for d in prices.sessions:
        o, c = prices.open_price(ticker, d), prices.close_price(ticker, d)
        if o is not None and c is not None:
            out[d] = (o, c)
    return out


def drift_labels(prices: ExecutionPrices, tickers: list[str], benchmark: str) -> pd.DataFrame:
    """Columns ticker, day, gap_ar, drift_ar, y (= drift_ar > 0); one row per labelled session."""
    market = _series(prices, benchmark)
    rows = []
    for t in tickers:
        stock = _series(prices, t)
        for d in prices.sessions:
            ar = abnormal_returns(d, stock, market, prices.sessions)
            if ar is not None:
                rows.append((t, d, ar.gap_ar, ar.drift_ar))
    df = pd.DataFrame(rows, columns=["ticker", "day", "gap_ar", "drift_ar"])
    df["y"] = (df["drift_ar"] > 0).astype(int)
    return df
