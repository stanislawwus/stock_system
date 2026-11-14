"""Streamlit client: Daily brief, Ticker, Ask, Strategy arena, Text analyzer, Evals.

Talks to the API over httpx only (AGENTS.md R13); never opens the database.
"""

from __future__ import annotations

import streamlit as st

from src.schemas import DISCLAIMER
from src.ui.client import ApiClient
from src.ui.screens import analyzer, arena, ask, brief, evals, ticker


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient()


def main() -> None:
    st.set_page_config(page_title="Stock Awareness Assistant", layout="wide")
    st.title("Stock Awareness Assistant")
    st.caption(f"⚠️ {DISCLAIMER}")
    client = get_client()
    if not client.health():
        st.error("API not reachable: run `make up` (or `make api`) and reload this page.")
        st.stop()

    def page(module, title: str, path: str, default: bool = False) -> st.Page:
        return st.Page(lambda: module.render(client), title=title, url_path=path, default=default)

    nav = st.navigation(
        [
            page(brief, "Daily brief", "brief", default=True),
            page(ticker, "Ticker", "ticker"),
            page(ask, "Ask", "ask"),
            page(arena, "Strategy arena", "arena"),
            page(analyzer, "Text analyzer", "analyzer"),
            page(evals, "Evals and data quality", "evals"),
        ]
    )
    nav.run()


main()
