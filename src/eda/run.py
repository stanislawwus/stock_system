"""`make eda`: figures into docs/figs and docs/02_data_understanding.md (CRISP-DM phase 2).

    uv run python -m src.eda.run

Development data only (train + validation for returns and labels; D-33). Definitions of gap_ar and
drift_ar: docs/02_architecture.md "Labels"; code: src/impact/labels.py. Wording follows R11: news
*coincides with* or *precedes* a move, nothing is said to cause one.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.collectors.common import ROOT, load_periods  # noqa: E402
from src.eda.data import decision_days, dev_cut, frame, news_frame, open_dev, sessions  # noqa: E402
from src.impact.labels import abnormal_returns  # noqa: E402
from src.marketview.clock import MARKET_TZ  # noqa: E402

FIGS = ROOT / "docs" / "figs"
DOC = ROOT / "docs" / "02_data_understanding.md"
BLUE, GREY, RED = "#2a6fb0", "#8a8f98", "#c0504d"
ITEM_GROUPS = ["2.02", "5.02", "1.01", "8.01", "7.01"]


def _style() -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e6e8eb",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "figure.dpi": 130,
            "font.size": 9,
        }
    )


def dev_sessions(con: duckdb.DuckDBPyConnection) -> list:
    """Sessions of train + validation only: the span returns and labels are studied on."""
    ps = load_periods(("train", "validation"))
    return [d for d in sessions(con) if any(p.contains(d) for p in ps)]


def _bars(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return frame(
        con,
        "SELECT ticker, bar_date, open, close FROM price_bars WHERE available_at < ? "
        "ORDER BY ticker, bar_date",
        [dev_cut()],
    )


def compute(con: duckdb.DuckDBPyConnection, benchmark: str = "SPY") -> dict[str, Any]:
    sess = dev_sessions(con)
    sess_set = set(sess)
    news = news_frame(con)
    et = pd.to_datetime(news["published_at"], utc=True).dt.tz_convert(MARKET_TZ)
    news = news.assign(et=et, day=et.dt.date, hour=et.dt.hour + et.dt.minute / 60)
    in_dev = news[news["day"].isin(sess_set)]
    per_td = in_dev.groupby(["ticker", "day"]).size().rename("n").reset_index()
    stocks = sorted(set(news["ticker"]) - {benchmark})
    per_ticker = per_td.groupby("ticker")["n"].sum().reindex(stocks, fill_value=0) / len(sess)

    bars = _bars(con)
    bars["bar_date"] = pd.to_datetime(bars["bar_date"]).dt.date
    px = {t: g.set_index("bar_date")[["open", "close"]] for t, g in bars.groupby("ticker")}
    stock_px = {t: {d: (r.open, r.close) for d, r in g.iterrows()} for t, g in px.items()}
    ret = []
    for t in stocks:
        g = px[t]
        prev = g["close"].shift()
        ret.append(
            pd.DataFrame(
                {
                    "ticker": t,
                    "close_close": g["close"] / prev - 1,
                    "overnight": g["open"] / prev - 1,
                    "intraday": g["close"] / g["open"] - 1,
                }
            )
            .dropna()
            .loc[lambda x: x.index.isin(sess_set)]
        )
    rets = pd.concat(ret)

    # labels: one per (ticker, decision day) with news; and per 8-K item group
    all_sessions = sessions(con)
    d_news = decision_days(news["published_at"], all_sessions)
    news = news.assign(D=d_news)
    spy = stock_px[benchmark]
    cache: dict = {}

    def label(t: str, d: Any) -> Any:
        if (t, d) not in cache:
            cache[(t, d)] = abnormal_returns(d, stock_px[t], spy, all_sessions)
        return cache[(t, d)]

    nd = news[news["D"].isin(sess_set) & news["ticker"].isin(stocks)][["ticker", "D"]]
    nd = nd.drop_duplicates()
    rows = []
    for t, d in zip(nd["ticker"], nd["D"], strict=True):
        r = label(t, d)
        if r:
            rows.append((t, d, r.gap_ar, r.drift_ar))
    lab = pd.DataFrame(rows, columns=["ticker", "D", "gap_ar", "drift_ar"])

    fil = frame(
        con,
        "SELECT ticker, items, accepted_at, available_at FROM filings WHERE form = '8-K' "
        "AND available_at < ?",
        [dev_cut()],
    )
    fil = fil.assign(D=decision_days(fil["available_at"], all_sessions))
    by_item: dict[str, pd.DataFrame] = {}
    for item in ITEM_GROUPS:
        sel = fil[fil["items"].map(lambda xs, i=item: i in list(xs)) & fil["D"].isin(sess_set)]
        out = []
        for t, d in zip(sel["ticker"], sel["D"], strict=True):
            r = label(t, d) if t in stock_px else None
            if r:
                out.append((r.gap_ar, r.drift_ar))
        by_item[item] = pd.DataFrame(out, columns=["gap_ar", "drift_ar"])
    no_news_days = _no_news_labels(stocks, sess_set, per_td, label, all_sessions)

    items = fil["items"].explode().value_counts()
    return {
        "sessions": len(sess),
        "news": news,
        "per_td": per_td,
        "per_ticker": per_ticker,
        "daily_total": in_dev.groupby("day").size(),
        "rets": rets,
        "labels": lab,
        "by_item": by_item,
        "no_news": no_news_days,
        "item_counts": items,
        "n_8k": len(fil),
        "benchmark": benchmark,
    }


def _no_news_labels(stocks, sess_set, per_td, label, all_sessions) -> pd.DataFrame:  # noqa: ANN001
    """Labels for ticker-days without any article (decision day = D), the comparison group."""
    have = set(zip(per_td["ticker"], per_td["day"], strict=True))
    rng = np.random.default_rng(0)
    days = sorted(sess_set)
    out = []
    for t in stocks:
        for d in rng.choice(np.array(days, dtype=object), size=min(20, len(days)), replace=False):
            if (t, d) in have:
                continue
            r = label(t, d)
            if r:
                out.append((r.gap_ar, r.drift_ar))
    return pd.DataFrame(out, columns=["gap_ar", "drift_ar"])


# --- figures -------------------------------------------------------------------------------------
def fig_news(r: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    counts = r["per_td"]["n"]
    ax[0].hist(np.clip(counts, 0, 60), bins=60, color=BLUE)
    ax[0].set_yscale("log")
    ax[0].set(
        title="Articles per ticker-day (clipped at 60)", xlabel="articles", ylabel="ticker-days"
    )
    pt = r["per_ticker"].sort_values(ascending=False)
    ax[1].bar(range(len(pt)), pt.values, color=BLUE, width=0.8)
    ax[1].set_xticks(range(0, len(pt), 8), [pt.index[i] for i in range(0, len(pt), 8)], rotation=60)
    ax[1].set(title="Mean articles per session, by ticker", ylabel="articles / session")
    dt = r["daily_total"]
    ax[2].plot(pd.to_datetime(dt.index), dt.values, color=BLUE, lw=1)
    ax[2].set(title="Articles per day, all tickers", ylabel="articles")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "news_per_ticker_day.png")
    plt.close(fig)


def fig_hours(r: dict[str, Any], out: Path) -> None:
    news = r["news"]
    wk = news[pd.to_datetime(news["et"]).dt.day_of_week < 5]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(wk["hour"], bins=np.arange(0, 24.5, 0.5), color=BLUE)
    ax.axvspan(9.5, 16, color=GREY, alpha=0.15, lw=0)
    ax.axvline(9.0, color=RED, lw=1.2)
    ax.text(8.9, ax.get_ylim()[1] * 0.93, "09:00 cutoff", color=RED, fontsize=8, ha="right")
    ax.text(9.6, ax.get_ylim()[1] * 0.80, "cash session 09:30-16:00", color="#555", fontsize=8)
    ax.set(
        title="Weekday publication time, New York (30-minute bins)",
        xlabel="hour (ET)",
        ylabel="articles",
        xticks=range(0, 25, 2),
        xlim=(0, 24),
    )
    fig.tight_layout()
    fig.savefig(out / "publication_hour.png")
    plt.close(fig)


def fig_returns(r: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for a, col, title in zip(
        ax,
        ("close_close", "overnight", "intraday"),
        ("Close to close", "Overnight (close D-1 to open D)", "Intraday (open D to close D)"),
        strict=True,
    ):
        x = r["rets"][col].clip(-0.1, 0.1)
        a.hist(x, bins=100, color=BLUE)
        a.set_yscale("log")
        a.set(title=title, xlabel="daily return (clipped at +/-10%)")
    fig.tight_layout()
    fig.savefig(out / "return_distributions.png")
    plt.close(fig)


def fig_gap_drift(r: dict[str, Any], out: Path) -> None:
    lab = r["labels"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hexbin(
        lab["gap_ar"].clip(-0.08, 0.08),
        lab["drift_ar"].clip(-0.08, 0.08),
        gridsize=50,
        bins="log",
        cmap="Blues",
        mincnt=1,
    )
    ax[0].axhline(0, color=GREY, lw=0.8)
    ax[0].axvline(0, color=GREY, lw=0.8)
    ax[0].set(
        title="gap_ar vs drift_ar, ticker-days with news",
        xlabel="gap_ar  (close D-1 to open D)",
        ylabel="drift_ar  (open D to close D+1)",
    )
    groups = {**{f"8-K {k}": v for k, v in r["by_item"].items()}, "news-days": lab,
              "no-news days": r["no_news"]}  # fmt: skip
    names = list(groups)
    x = np.arange(len(names))
    w = 0.38
    for off, col, colr in ((-w / 2, "gap_ar", BLUE), (w / 2, "drift_ar", GREY)):
        m = [groups[n][col].mean() * 100 for n in names]
        se = [groups[n][col].std() / np.sqrt(max(len(groups[n]), 1)) * 100 for n in names]
        ax[1].bar(x + off, m, w, yerr=se, color=colr, label=col, error_kw={"lw": 0.8})
    ax[1].set_xticks(x, [f"{n}\nn={len(groups[n])}" for n in names], fontsize=7)
    ax[1].axhline(0, color="#555", lw=0.8)
    ax[1].legend(frameon=False)
    ax[1].set(title="Mean abnormal return by event group (+/- 1 s.e.)", ylabel="percent")
    fig.tight_layout()
    fig.savefig(out / "gap_vs_drift.png")
    plt.close(fig)


# --- document ------------------------------------------------------------------------------------
def _p(x: float) -> str:
    return f"{x:.2%}"


def stats(r: dict[str, Any]) -> dict[str, Any]:
    rets, lab = r["rets"], r["labels"]
    news = r["news"]
    wk = news[pd.to_datetime(news["et"]).dt.day_of_week < 5]
    s: dict[str, Any] = {
        "sessions": r["sessions"],
        "ticker_days_with_news": int(len(r["per_td"])),
        "median_per_ticker_day": float(r["per_td"]["n"].median()),
        "p99_per_ticker_day": float(r["per_td"]["n"].quantile(0.99)),
        "top_tickers": {k: float(v) for k, v in r["per_ticker"].nlargest(3).items()},
        "bottom_tickers": {k: float(v) for k, v in r["per_ticker"].nsmallest(3).items()},
        "wk_pre_open": float((wk["hour"] < 9.5).mean()),
        "wk_before_cutoff": float((wk["hour"] < 9.0).mean()),
        "wk_session": float(((wk["hour"] >= 9.5) & (wk["hour"] < 16)).mean()),
        "wk_after": float((wk["hour"] >= 16).mean()),
        "peak_hour": int(wk["hour"].astype(int).value_counts().idxmax()),
        "n_labels": int(len(lab)),
        "corr_gap_drift": float(lab["gap_ar"].corr(lab["drift_ar"])),
    }
    for col in ("close_close", "overnight", "intraday"):
        x = rets[col]
        s[col] = {
            "std": float(x.std()),
            "kurtosis": float(x.kurt()),
            "p01": float(x.quantile(0.01)),
            "p99": float(x.quantile(0.99)),
        }
    s["abs_gap_news"] = float(lab["gap_ar"].abs().mean())
    s["abs_gap_nonews"] = float(r["no_news"]["gap_ar"].abs().mean())
    s["abs_drift_news"] = float(lab["drift_ar"].abs().mean())
    s["abs_drift_nonews"] = float(r["no_news"]["drift_ar"].abs().mean())
    s["by_item"] = {
        k: {
            "n": int(len(v)),
            "gap": float(v["gap_ar"].mean()),
            "abs_gap": float(v["gap_ar"].abs().mean()),
            "drift": float(v["drift_ar"].mean()),
            "drift_se": float(v["drift_ar"].std() / np.sqrt(max(len(v), 1))),
            "abs_drift": float(v["drift_ar"].abs().mean()),
        }
        for k, v in r["by_item"].items()
    }
    s["item_counts"] = {k: int(v) for k, v in r["item_counts"].head(8).items()}
    s["n_8k"] = r["n_8k"]
    s["n_nonews"] = int(len(r["no_news"]))
    far = {k: v for k, v in s["by_item"].items() if abs(v["drift"]) > 2 * v["drift_se"]}
    s["drift_note"] = (
        "groups whose mean lies more than 2 standard errors from zero: "
        + ", ".join(f"{k} ({v['drift'] * 100:+.2f} %, n={v['n']})" for k, v in far.items())
        + "."
        if far
        else "every group's mean is within 2 standard errors of zero."
    )
    return s


def render(s: dict[str, Any], sha: str, quality_summary: str) -> str:
    items = ", ".join(f"{k} {v}" for k, v in s["item_counts"].items())
    rows = "\n".join(
        f"| {k} | {v['n']} | {v['gap'] * 100:+.2f} | {v['abs_gap'] * 100:.2f} | "
        f"{v['drift'] * 100:+.2f} (s.e. {v['drift_se'] * 100:.2f}) | {v['abs_drift'] * 100:.2f} |"
        for k, v in s["by_item"].items()
    )
    return f"""# 02 — Data understanding (EDA)

Generated by `make eda` (`src/eda/run.py`, git `{sha}`); do not edit by hand. CRISP-DM phase 2, the
gate for data preparation. Figures: `docs/figs/`. Scope: train + validation sessions ({s["sessions"]}
sessions) for returns and labels, never the test period (R3, D-33). Coverage per source and the
list of holidays: `docs/02_data_sources.md`; rule-by-rule data quality (duplicates, missing days,
UTC, headline length, language, tickers, 8-K headers): `docs/04_data_quality.md`.
{quality_summary}
Wording (R11): an article or filing *coincided with* or *preceded* a move; none of these tables
makes a causal claim. `gap_ar` and `drift_ar` are defined in `docs/02_architecture.md`:
`gap_ar` = abnormal return close(D-1) to open(D) (already realised when we enter; diagnostic only),
`drift_ar` = abnormal return open(D) to close(D+1) (the tradable target); abnormal = minus SPY.

## 1. News per ticker per day

![news per ticker-day](figs/news_per_ticker_day.png)

- {s["ticker_days_with_news"]:,} ticker-days carry at least one article; median {s["median_per_ticker_day"]:.0f},
  99th percentile {s["p99_per_ticker_day"]:.0f} articles. The distribution is heavy-tailed (log axis).
- Busiest names (articles per session): {", ".join(f"{k} {v:.1f}" for k, v in s["top_tickers"].items())};
  quietest: {", ".join(f"{k} {v:.1f}" for k, v in s["bottom_tickers"].items())}. A raw article count is
  therefore a statement about the ticker's usual attention, not about the day: any news-count feature
  must be normalised per ticker (e.g. z-score against its own trailing window).

## 2. Publication hour against the market open

![publication hour](figs/publication_hour.png)

- On weekdays {_p(s["wk_before_cutoff"])} of articles appear before the 09:00 ET cutoff, {_p(s["wk_pre_open"])} before
  the 09:30 open, {_p(s["wk_session"])} during the session and {_p(s["wk_after"])} after the close; the busiest hour
  is {s["peak_hour"]}:00 ET.
- Consequence for the time rules: only the before-cutoff share (plus the previous evening and the weekend)
  is usable at the open of day D; session articles can only be used at the next open. This is why
  `decision_day` (first session whose cutoff is at or after `available_at`) is computed per item,
  and why a same-day return is never paired with same-day news in a signal.

## 3. Return distributions

![return distributions](figs/return_distributions.png)

| Return | Std | 1st pct | 99th pct | Excess kurtosis |
|---|---|---|---|---|
| close to close | {_p(s["close_close"]["std"])} | {_p(s["close_close"]["p01"])} | {_p(s["close_close"]["p99"])} | {s["close_close"]["kurtosis"]:.1f} |
| overnight (close D-1 to open D) | {_p(s["overnight"]["std"])} | {_p(s["overnight"]["p01"])} | {_p(s["overnight"]["p99"])} | {s["overnight"]["kurtosis"]:.1f} |
| intraday (open D to close D) | {_p(s["intraday"]["std"])} | {_p(s["intraday"]["p01"])} | {_p(s["intraday"]["p99"])} | {s["intraday"]["kurtosis"]:.1f} |

- All three are fat-tailed (kurtosis well above 0), so mean-based metrics and Gaussian assumptions need
  care; the impact eval reports direction accuracy and AUC rather than R².
- Overnight and intraday moves are of comparable size, so the news published overnight can already be
  partly reflected at the open: this is exactly what `gap_ar` measures.

## 4. gap_ar against drift_ar

![gap vs drift](figs/gap_vs_drift.png)

Events: {s["n_labels"]:,} (ticker, decision day) pairs with at least one article in train + validation, and
{s["n_8k"]:,} 8-Ks (item counts: {items}). Means in percent; the 8-K rows are filings whose decision day lies in
train + validation.

| Group | n | mean gap_ar | mean abs gap_ar | mean drift_ar | mean abs drift_ar |
|---|---|---|---|---|---|
{rows}
| news-days | {s["n_labels"]} | | {s["abs_gap_news"] * 100:.2f} | | {s["abs_drift_news"] * 100:.2f} |
| no-news days (random sample) | {s["n_nonews"]} | | {s["abs_gap_nonews"] * 100:.2f} | | {s["abs_drift_nonews"] * 100:.2f} |

- Correlation of gap_ar and drift_ar over news-days: {s["corr_gap_drift"]:+.3f}.
- The mean absolute `gap_ar` on news-days is {s["abs_gap_news"] * 100:.2f} % against
  {s["abs_gap_nonews"] * 100:.2f} % on randomly drawn no-news days (same tickers): days with articles
  coincide with larger opening gaps. After the open, the mean absolute `drift_ar` is
  {s["abs_drift_news"] * 100:.2f} % against {s["abs_drift_nonews"] * 100:.2f} %.
- Signed `drift_ar` means: {s["drift_note"]} Five groups are tested at once, so one interval
  excluding zero is not strong evidence; the 8-K groups are also small. Any strategy claim has to
  be settled by the arena on `validation`, not by this table.

## Conclusions for the next phase

1. Normalise attention features per ticker; use `decision_day`, not the calendar date, to pair text with prices.
2. Keep headlines flagged by `clean.check_headline` and roundup articles out of labelling sheets and sentiment indices.
3. Strip item headers from 8-K text before classification; the item numbers are the gold labels.
4. Treat the opening gap as largely realised when the strategy enters (`gap_ar`), and judge value on `drift_ar` only.
"""


def main() -> int:
    _style()
    FIGS.mkdir(parents=True, exist_ok=True)
    con = open_dev()
    r = compute(con)
    for fn in (fig_news, fig_hours, fig_returns, fig_gap_drift):
        fn(r, FIGS)
    s = stats(r)
    (FIGS / "eda_stats.json").write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    ).stdout.strip()
    DOC.write_text(render(s, sha, ""), encoding="utf-8", newline="\n")
    print(f"wrote {DOC.relative_to(ROOT)} and figures in {FIGS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
