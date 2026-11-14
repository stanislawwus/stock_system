"""Daily brief: one row per watchlist stock as of the 09:00 ET cutoff of day D."""

from __future__ import annotations

from datetime import date

import streamlit as st

from src.ui import components as ui
from src.ui.client import ApiClient
from src.ui.views import brief_table, by_component, sample_status


def render(client: ApiClient) -> None:
    ui.page_header(
        "Daily brief",
        "Watchlist at the 09:00 ET cutoff of the decision day: everything shown was public "
        "before it. News and filings coincided with moves; they are not shown to explain them.",
    )
    day = ui.trading_day_input()
    brief = ui.fetch("Building the brief...", lambda: client.brief(day or None))
    if brief is None:
        ui.page_footer()
        return
    if not brief.items:
        ui.empty("The watchlist is empty: nothing to brief. Add names in config/watchlist.yaml.")
        ui.page_footer()
        return

    st.subheader(f"As of {brief.as_of:%Y-%m-%d %H:%M} UTC")
    table = brief_table(brief)
    st.dataframe(table, hide_index=True, width="stretch")
    if table["sentiment index"].isna().all():
        ui.empty("No scored news in the decision window for any name (sentiment index blank).")

    strategies = sorted({i.signal.strategy for i in brief.items if i.signal})
    evals = ui.fetch("Checking in-sample status...", client.evals)
    if strategies and evals is not None:
        day_d = brief.as_of.date() if not day else date.fromisoformat(day)
        for s in strategies:
            flag, text = sample_status(day_d, s, by_component(evals))
            ui.in_sample_banner(flag, text)
    st.caption(
        "Score in [-1, 1]; score <= 0 means cash (long-only). Sentiment index = mean FinBERT "
        "score of the relevant, de-duplicated news in the decision window; blank = no news."
    )
    ui.page_footer()
