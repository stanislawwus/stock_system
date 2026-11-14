"""Pure helpers that turn API responses into tables for the Streamlit page (no I/O, testable)."""

from __future__ import annotations

import pandas as pd

from src.schemas import EvalResult, SignalsResponse


def signals_table(resp: SignalsResponse) -> pd.DataFrame:
    """One row per strategy: score, position it implies (long-only), reasons."""
    rows = [
        {
            "strategy": s.strategy,
            "score": round(s.score, 3),
            "position": "long" if s.score > 0 else "cash",
            "reasons": ", ".join(s.reasons),
        }
        for s in resp.signals
    ]
    return pd.DataFrame(rows, columns=["strategy", "score", "position", "reasons"])


def equity_frame(rec: EvalResult) -> pd.DataFrame:
    """Equity curve stored in a backtest eval record's details, indexed by date."""
    curve = (rec.details or {}).get("equity_curve") or []
    df = pd.DataFrame(curve, columns=["date", "equity"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def backtest_summary(rec: EvalResult) -> dict[str, float | int | None]:
    d = rec.details or {}
    return {
        "Sharpe": rec.value,
        "buy-and-hold Sharpe": rec.target,
        "total return": d.get("total_return"),
        "max drawdown": d.get("max_drawdown"),
        "trades": d.get("n_trades"),
    }
