"""Shared Streamlit pieces: the call wrapper with loading / error / degraded states, the
disclaimer, and the in-sample banner. Pages never handle httpx errors themselves."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import streamlit as st

from src.schemas import DISCLAIMER
from src.ui.client import ApiClient, ApiError, ApiTimeout, ApiUnreachable

T = TypeVar("T")

UNREACHABLE = "The API is not reachable. Start it with `make up` (or `make api`) and reload."
TIMEOUT = "The API did not answer in time (a model may still be loading). Try again in a minute."
DEGRADED = (
    "Degraded: the API answered that this feature or its data is not available right now "
    "({detail}). Other pages keep working."
)


def disclaimer() -> None:
    st.caption(f"⚠️ {DISCLAIMER} Signals are suggestions with a confidence, not predictions.")


def fetch(label: str, call: Callable[[], T]) -> T | None:
    """Run an API call under a spinner (loading state); show the error or degraded state and
    return None when it fails, so the caller only handles the success path and the empty case."""
    try:
        with st.spinner(label):
            return call()
    except ApiTimeout:
        st.error(TIMEOUT)
    except ApiUnreachable:
        st.error(UNREACHABLE)
    except ApiError as exc:
        if exc.degraded:
            st.warning(DEGRADED.format(detail=exc.detail))
        else:
            st.error(f"The API refused the request ({exc.status}): {exc.detail}")
    return None


def empty(message: str) -> None:
    st.info(message)


def in_sample_banner(flag: bool | None, text: str) -> None:
    """In-sample numbers are not evidence of skill: say so next to them."""
    if flag is True:
        st.warning(text)
    elif flag is False:
        st.caption(f"Out of sample: {text}")
    else:
        st.caption(f"In-sample status unknown: {text}")


def page_header(title: str, intro: str) -> None:
    st.header(title)
    st.caption(intro)


def page_footer() -> None:
    st.divider()
    disclaimer()


def trading_day_input(
    label: str = "Decision day D (YYYY-MM-DD; empty = last validation day)",
) -> str:
    return st.text_input(label, value="", max_chars=10).strip()


def load_evals(client: ApiClient):
    """All stored eval records, or None with the state already shown."""
    return fetch("Loading stored evals...", client.evals)
