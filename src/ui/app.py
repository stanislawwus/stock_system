"""Streamlit client v0: ticker -> today's strategy signals + finbert_v0 equity curve on validation.

Talks to the API over httpx only (AGENTS.md R13); never opens the database.
"""

from __future__ import annotations

import httpx
import streamlit as st

from src.schemas import DISCLAIMER
from src.ui.client import ApiClient
from src.ui.views import backtest_summary, equity_frame, signals_table

BACKTEST = "backtest_finbert_v0_validation"

st.set_page_config(page_title="Stock Awareness Assistant", layout="wide")
st.title("Stock Awareness Assistant")
st.caption(DISCLAIMER)

client = ApiClient()
if not client.health():
    st.error("API not reachable: run `make up` (or `make api`).")
    st.stop()

left, right = st.columns([1, 1])
ticker = left.text_input("Ticker", value="AAPL", max_chars=10).strip().upper()
day = right.text_input(
    "Decision day D (YYYY-MM-DD, empty = last validation day)", value="", max_chars=10
).strip()

if ticker:
    try:
        resp = client.signals(ticker, day or None)
    except httpx.HTTPStatusError as exc:
        st.warning(f"{exc.response.status_code}: {exc.response.json().get('detail')}")
    else:
        st.subheader(f"Signals for {resp.ticker} as of {resp.as_of:%Y-%m-%d %H:%M} UTC")
        st.dataframe(signals_table(resp), hide_index=True, width="stretch")
        st.caption(
            "Score in [-1, 1]; score <= 0 means cash (long-only). Signals are suggestions "
            "with a confidence, computed from data public before 09:00 ET of D. " + resp.disclaimer
        )

st.subheader("finbert_v0 on the validation split (watchlist, 5 bps per side)")
records = client.evals(BACKTEST)
if records:
    rec = records[0]
    st.line_chart(equity_frame(rec), y="equity")
    cols = st.columns(5)
    for col, (k, v) in zip(cols, backtest_summary(rec).items(), strict=True):
        col.metric(k, "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else v))
    st.caption(rec.notes)
else:
    st.info(f"No {BACKTEST} result yet: run `make backtest STRATEGY=finbert_v0 SPLIT=val`.")

st.caption(DISCLAIMER)
