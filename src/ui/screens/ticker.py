"""Ticker page: price, every strategy's signal, SEC filings with events and facts, news."""

from __future__ import annotations

from datetime import date

import streamlit as st

from src.ui import components as ui
from src.ui.client import ApiClient
from src.ui.views import (
    by_component,
    facts_table,
    filings_table,
    news_markdown,
    sample_status,
    signals_table,
)

FALLBACK_TICKER = "AAPL"


def render(client: ApiClient) -> None:
    ui.page_header(
        "Ticker",
        "Price, strategy signals, SEC filings and cited news as of the cutoff of day D. News "
        "and filings are listed because they were published before the cutoff; the page does "
        "not claim they moved the price.",
    )
    wl = ui.fetch("Loading the watchlist...", client.watchlist)
    names = [i.ticker for i in wl.items if i.kind == "stock"] if wl else []
    left, right = st.columns(2)
    if names:
        ticker = left.selectbox("Ticker", names)
    else:
        ticker = left.text_input("Ticker", value=FALLBACK_TICKER, max_chars=10).strip().upper()
    with right:
        day = ui.trading_day_input()
    if not ticker:
        ui.empty("Pick a ticker.")
        ui.page_footer()
        return
    as_of = day or None

    st.subheader("Price")
    prices = ui.fetch("Loading prices...", lambda: client.prices(ticker, as_of))
    if prices is not None:
        if prices.bars:
            st.line_chart({"close": {b.bar_date: b.close for b in prices.bars}})
            st.caption(
                f"Daily closes up to {prices.bars[-1].bar_date} (the last bar closed before the "
                f"{prices.as_of:%Y-%m-%d %H:%M} UTC cutoff)."
            )
        else:
            ui.empty(f"No price bars for {ticker} before this cutoff.")

    st.subheader("Signals")
    sig = ui.fetch("Computing signals...", lambda: client.signals(ticker, as_of))
    if sig is not None:
        if sig.signals:
            st.dataframe(signals_table(sig), hide_index=True, width="stretch")
            evals = ui.fetch("Checking in-sample status...", client.evals)
            if evals is not None:
                d = sig.as_of.date() if not day else date.fromisoformat(day)
                for s in sig.signals:
                    flag, text = sample_status(d, s.strategy, by_component(evals))
                    if flag is not False:
                        ui.in_sample_banner(flag, f"{s.strategy}: {text}")
        else:
            ui.empty("No strategy produced a signal for this ticker and day.")
        st.caption("Score <= 0 means cash. A suggestion with a confidence, not a prediction.")

    st.subheader("SEC filings, events and facts")
    ev = ui.fetch("Loading filings...", lambda: client.events(ticker, as_of))
    if ev is not None:
        if ev.filings:
            st.dataframe(
                filings_table(ev),
                hide_index=True,
                width="stretch",
                column_config={"link": st.column_config.LinkColumn("link")},
            )
        else:
            ui.empty("No filings in the last 90 days before this cutoff.")
        if ev.facts:
            with st.expander(f"Facts (XBRL, {len(ev.facts)})"):
                st.dataframe(facts_table(ev), hide_index=True, width="stretch")

    st.subheader("News")
    news = ui.fetch("Loading news...", lambda: client.news(ticker, as_of))
    if news is not None:
        if news.items:
            st.markdown("\n".join(news_markdown(news)))
            st.caption("Each line cites the article: link, source, UTC availability time, id.")
        else:
            ui.empty("No news for this ticker in the last 7 days before the cutoff.")
    ui.page_footer()
