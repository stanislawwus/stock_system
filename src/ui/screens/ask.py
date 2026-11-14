"""Ask: question -> answer with citations from POST /chat. Shows a degraded state while the chat
agent is not implemented (the API answers 501) or its data is unavailable (503)."""

from __future__ import annotations

import streamlit as st

from src.ui import components as ui
from src.ui.client import ApiClient

HISTORY_KEY = "ask_history"
MAX_QUESTION = 2000  # mirrors ChatRequest.question max_length


def render(client: ApiClient) -> None:
    ui.page_header(
        "Ask",
        "Questions about the watchlist, answered from stored news, filings and prices, with "
        "citations. News is described as published before or on the same day as a move, never "
        "as its cause.",
    )
    history: list[dict] = st.session_state.setdefault(HISTORY_KEY, [])
    question = st.text_area("Your question", max_chars=MAX_QUESTION, height=100)
    if st.button("Ask", disabled=not question.strip()):
        resp = ui.fetch("Asking...", lambda: client.chat(question.strip()))
        if resp is not None:
            history.insert(0, {"q": question.strip(), "r": resp})
    if not history:
        ui.empty("No questions yet. Try: What happened to NVDA this week?")
    for item in history:
        r = item["r"]
        with st.container(border=True):
            st.markdown(f"**Q:** {item['q']}")
            st.markdown(r.answer)
            st.caption(f"method: {r.method}")
            if r.citations:
                for c in r.citations:
                    st.markdown(
                        f"- `{c.kind}` `{c.source_ref}` · available {c.available_at:%Y-%m-%d %H:%M}"
                        f" UTC · “{c.quote}”"
                    )
            else:
                st.warning("This answer carries no citations: treat it as unsupported.")
            st.caption(r.disclaimer)
    ui.page_footer()
