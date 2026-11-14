"""Evals and data quality: every stored eval with target, value, CI and the in-sample flag."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui import components as ui
from src.ui.client import ApiClient
from src.ui.views import evals_table

QUALITY_COMPONENT = "collectors"


def render(client: ApiClient) -> None:
    ui.page_header(
        "Evals and data quality",
        "Stored results of `make eval`, `make backtest` and `make arena`. A score below its "
        "target is shown as is; IN-SAMPLE marks values fitted or chosen on the data they report.",
    )
    recs = ui.load_evals(client)
    if recs is None:
        ui.page_footer()
        return
    if not recs:
        ui.empty("No eval results stored yet: run `make eval`.")
        ui.page_footer()
        return

    hide_bt = st.checkbox("Hide per-strategy backtest records", value=True)
    shown = [r for r in recs if not (hide_bt and r.component.startswith("backtest_"))]
    table = evals_table(shown)
    st.dataframe(table, hide_index=True, width="stretch")
    below = int((table["meets target"] == "no").sum())
    n_in = int((table["in-sample"] == "IN-SAMPLE").sum())
    st.caption(
        f"{len(table)} records; {below} below target; {n_in} in-sample. CIs appear only where "
        "the eval stored one (bootstrap); n/a means none was computed."
    )

    pick = st.selectbox("Details", [r.component for r in shown])
    rec = next(r for r in shown if r.component == pick)
    st.markdown(f"**Why this metric:** {rec.why_this_metric}")
    st.markdown(f"**Gold set:** {rec.gold_set}")
    if rec.in_sample:
        st.warning(f"IN-SAMPLE: {rec.in_sample_note}")
    if rec.notes:
        st.markdown(f"**Notes:** {rec.notes}")

    st.subheader("Data quality")
    dq = next((r for r in recs if r.component == QUALITY_COMPONENT), None)
    if dq is None or not dq.details:
        ui.empty("No collector quality record stored: run `make collect-dev` then `make eval`.")
    else:
        scalars = {k: v for k, v in dq.details.items() if isinstance(v, int | float | str)}
        st.caption(f"{dq.metric} = {dq.value} (target {dq.target}); {dq.notes}")
        st.dataframe(
            pd.DataFrame({"check": list(scalars), "value": [str(v) for v in scalars.values()]}),
            hide_index=True,
            width="stretch",
        )

    st.subheader("LLM models and today's budget")
    models = ui.fetch("Loading models...", client.models)
    if models:
        st.dataframe(
            pd.DataFrame([m.model_dump() for m in models]), hide_index=True, width="stretch"
        )
    elif models is not None:
        ui.empty("No models configured.")
    ui.page_footer()
