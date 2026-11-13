"""Streamlit client. Pages are added as their endpoints land; this shell only checks the API."""

from __future__ import annotations

import streamlit as st

from src.schemas import DISCLAIMER
from src.ui.client import ApiClient

st.set_page_config(page_title="Stock Awareness Assistant", layout="wide")
st.title("Stock Awareness Assistant")
st.caption(DISCLAIMER)

client = ApiClient()
st.write("API reachable" if client.health() else "API not reachable: run `make api`")
