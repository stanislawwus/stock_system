"""Strategy arena: leaderboard with confidence intervals, equity curves, simulator replay."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui import components as ui
from src.ui.client import ApiClient
from src.ui.views import arena_rows, by_component, equity_curves, num, replay_table

SPLIT = "validation"


def render(client: ApiClient) -> None:
    ui.page_header(
        "Strategy arena",
        "Every strategy on the validation split through the simulator (open of D, 5 bps per "
        "side, long-only). Results are replayed history of a simulation, not advice.",
    )
    evals = ui.load_evals(client)
    if evals is None:
        ui.page_footer()
        return
    recs = by_component(evals)
    arena = recs.get("arena")
    if arena is None or not (arena.details or {}).get("leaderboard"):
        ui.empty("No stored arena result yet: run `make arena`.")
        ui.page_footer()
        return
    det = arena.details or {}

    if st.button("Refresh leaderboard from the stored result (POST /arena)"):
        job = ui.fetch("Running the arena job...", lambda: client.run_arena(SPLIT))
        if job is not None and job.status == "failed":
            st.error(f"Arena job failed: {job.error}")
        elif job is not None and job.status != "done":
            st.warning("The job is still running; press the button again in a moment.")
        elif job is not None and job.result:
            st.success(f"Loaded {job.result['source']} (recomputed: {job.result['recomputed']}).")
            det = {**det, "leaderboard": job.result["leaderboard"]}

    st.subheader(f"Leaderboard, {det.get('start')} to {det.get('end')} ({det.get('n_days')} days)")
    board = arena_rows(det)
    st.dataframe(board, hide_index=True, width="stretch")
    if (board["in-sample on validation"] == "IN-SAMPLE").any():
        st.warning(
            "IN-SAMPLE rows chose their calibrator and hold/threshold on val_a, the first half of "
            "validation: their full-validation Sharpe is partly fitted. Read the val_b columns."
        )
    st.caption(
        "Sharpe CIs are 95% moving-block bootstrap intervals over a few dozen days: wide, and "
        "most rows overlap each other and zero. "
        f"Headline: {arena.metric} = {num(arena.value)} vs target {num(arena.target, 2)}."
    )
    ui.in_sample_banner(arena.in_sample, arena.in_sample_note or arena.notes)

    st.subheader("Equity curves (rebased to 1.0)")
    names = [r["strategy"] for r in det["leaderboard"]]
    chosen = st.multiselect("Strategies", names, default=names[:4])
    curves = equity_curves(
        [recs[f"backtest_{n}_{SPLIT}"] for n in names if f"backtest_{n}_{SPLIT}" in recs], chosen
    )
    missing = [n for n in chosen if n not in curves.columns]
    if missing:
        st.caption(f"No stored backtest curve for: {', '.join(missing)}.")
    if curves.empty:
        ui.empty("Pick at least one strategy with a stored backtest.")
        ui.page_footer()
        return
    st.line_chart(curves)

    st.subheader("Simulator replay")
    days = list(curves.index)
    upto = st.select_slider(
        "Replay up to", options=days, value=days[-1], format_func=lambda d: f"{d:%Y-%m-%d}"
    )
    table = replay_table(curves, pd.Timestamp(upto))
    st.line_chart(curves[curves.index <= upto])
    st.dataframe(
        table.style.format({"cumulative return": "{:+.2%}", "max drawdown so far": "{:.2%}"}),
        hide_index=True,
        width="stretch",
    )
    st.caption("Replay of stored simulator output; nothing is recomputed or traded.")
    ui.page_footer()
