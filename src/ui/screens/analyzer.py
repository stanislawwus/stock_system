"""Text analyzer: paste any text, get sentiment (events and facts when the API provides them)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui import components as ui
from src.ui.client import ApiClient

MAX_TEXT = 20000  # mirrors AnalyzeRequest.text max_length


def render(client: ApiClient) -> None:
    ui.page_header(
        "Text analyzer",
        "Paste a headline or paragraph. The text is treated as data: instructions inside it are "
        "ignored by the analysis.",
    )
    text = st.text_area("Text", max_chars=MAX_TEXT, height=160)
    ticker = st.text_input("Ticker (optional)", max_chars=10).strip().upper()
    if st.button("Analyze", disabled=not text.strip()):
        res = ui.fetch("Analyzing...", lambda: client.analyze(text.strip(), ticker or None))
        if res is not None:
            s = res.sentiment
            word = {1: "positive", 0: "neutral", -1: "negative"}[s.label]
            a, b, c = st.columns(3)
            a.metric("Sentiment", word)
            b.metric("Score", f"{s.score:+.3f}")
            c.metric("Method", res.method)
            if res.events is None:
                ui.empty("No event label: the event classifier does not run on free text yet.")
            else:
                st.markdown(
                    f"**Events:** {', '.join(res.events.labels)} "
                    f"(confidence {res.events.confidence:.2f}, {res.events.method})"
                )
            if res.facts:
                st.dataframe(
                    pd.DataFrame([f.model_dump(mode="json") for f in res.facts]),
                    hide_index=True,
                    width="stretch",
                )
            else:
                ui.empty("No financial facts found in this text.")
            st.caption(res.disclaimer)
    else:
        ui.empty("Enter some text and press Analyze.")
    ui.page_footer()
