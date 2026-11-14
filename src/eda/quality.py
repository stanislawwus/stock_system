"""Data-quality audit of the development data -> docs/04_data_quality.md (CRISP-DM phase 2/3).

    uv run python -m src.eda.quality            # prints the audit, writes docs/04_data_quality.md

`audit()` returns plain numbers (tested on a small seeded database); `render()` turns them into the
document. Only rows available before the first test cutoff are read (D-33).
"""

from __future__ import annotations

import argparse
import re
from typing import Any

import duckdb
import pandas as pd

from src.collectors.common import ROOT, load_periods, load_universe
from src.collectors.quality import price_completeness
from src.eda.data import decision_days, dev_cut, frame, news_frame, open_dev, sessions
from src.marketview.clock import market_phase
from src.nlp import clean

DOC = ROOT / "docs" / "04_data_quality.md"
PERIODS = ("warmup", "train", "validation")
_ITEM_HEADING = re.compile(r"(?im)^\s*item\s+\d{1,2}\.\d{2}\b")


def _pct(x: float) -> str:
    return f"{x:.2%}"


# --- sections ------------------------------------------------------------------------------------
def duplicates(news: pd.DataFrame) -> dict[str, Any]:
    cfg = clean.config()["dedup"]
    n = len(news)
    articles = news["id"].nunique()
    per_article = news.groupby("id")["ticker"].nunique()
    df = news[["ticker", "headline", "published_at", "id"]].copy()
    df["key"] = df["headline"].map(clean.dedup_key)
    df = df.sort_values(["ticker", "key", "published_at"])
    grp = df.groupby(["ticker", "key"])
    gap = grp["published_at"].diff()
    window = pd.Timedelta(hours=int(cfg["window_hours"]))
    repeat_any = grp["id"].transform("size") > 1
    syndicated = (gap <= window) & (df["id"] != grp["id"].shift())
    return {
        "rows": n,
        "articles": int(articles),
        "pk_duplicates": int(n - len(news.drop_duplicates(["id", "ticker"]))),
        "multi_ticker_articles": int((per_article > 1).sum()),
        "multi_ticker_share": float((per_article > 1).mean()),
        "same_ticker_same_headline_rows": int(repeat_any.sum()),
        "syndicated_copies": int(syndicated.sum()),
        "syndicated_share": float(syndicated.sum() / n) if n else 0.0,
        "window_hours": int(cfg["window_hours"]),
    }


def missing_days(
    con: duckdb.DuckDBPyConnection, news: pd.DataFrame, tickers: list[str], benchmark: str
) -> dict[str, Any]:
    periods = [p for p in load_periods(PERIODS)]
    comp = price_completeness(con, tickers, benchmark, periods)
    all_sessions = sessions(con, benchmark)
    have = set(all_sessions)
    span = pd.bdate_range(min(have), max(have)).date
    non_sessions = [d for d in span if d not in have]
    below = {t: v for t, v in comp.items() if v["completeness"] < 1.0}
    bad = con.execute(
        "SELECT count(*) FROM price_bars WHERE available_at < ? AND "
        "(open <= 0 OR close <= 0 OR high < low OR volume < 0)",
        [dev_cut()],
    ).fetchone()[0]
    flat = con.execute(
        "SELECT count(*) FROM price_bars WHERE available_at < ? AND ticker <> ? AND volume = 0",
        [dev_cut(), benchmark],
    ).fetchone()[0]
    # news coverage per ticker-session, train + validation sessions only
    dev = [p for p in load_periods(("train", "validation"))]
    sess = [d for d in all_sessions if any(p.contains(d) for p in dev)]
    news = news.assign(day=pd.to_datetime(news["published_at"]).dt.tz_convert("America/New_York"))
    news = news.assign(day=news["day"].dt.date)
    pairs = news[news["day"].isin(sess)].groupby(["ticker", "day"]).size()
    stocks = [t for t in tickers if t != benchmark]
    expected = len(sess) * len(stocks)
    per_ticker = pairs.groupby(level=0).size().reindex(stocks, fill_value=0) / max(len(sess), 1)
    return {
        "sessions": {p.name: len([d for d in all_sessions if p.contains(d)]) for p in periods},
        "weekdays_without_session": [d.isoformat() for d in non_sessions],
        "tickers_below_full": {
            t: {"present": v["present"], "expected": v["expected"], "missing": v["missing"][:10]}
            for t, v in below.items()
        },
        "min_completeness": min(v["completeness"] for v in comp.values()),
        "bad_bars": int(bad),
        "zero_volume_bars": int(flat),
        "news_ticker_days": expected,
        "news_coverage_lowest": {t: float(v) for t, v in per_ticker.nsmallest(3).items()},
        "news_per_ticker_day_median": float(pairs.median()) if len(pairs) else 0.0,
        "news_ticker_days_with_news": int(len(pairs)),
        "news_ticker_day_coverage": len(pairs) / expected if expected else 0.0,
    }


def utc_and_open(con: duckdb.DuckDBPyConnection, news: pd.DataFrame) -> dict[str, Any]:
    naive = con.execute(
        "SELECT count(*) FROM information_schema.columns WHERE column_name IN "
        "('available_at', 'published_at', 'accepted_at') AND data_type <> 'TIMESTAMP WITH TIME ZONE'"
    ).fetchone()[0]
    offsets = {str(ts.utcoffset()) for ts in news["published_at"].head(1000)}
    ts = pd.to_datetime(news["published_at"], utc=True)
    et = ts.dt.tz_convert("America/New_York")
    phase = ts.map(lambda t: market_phase(t.to_pydatetime()))
    weekend = et.dt.day_of_week >= 5
    sess = sessions(con)
    d = decision_days(news["published_at"], sess)
    news_day = et.dt.date
    lag = (pd.to_datetime(d.dropna()) - pd.to_datetime(news_day[d.notna()])).dt.days
    return {
        "non_timestamptz_columns": int(naive),
        "returned_utc_offsets": sorted(offsets),
        "published_min": str(ts.min()),
        "published_max": str(ts.max()),
        "phase_share": {k: float(v) for k, v in phase.value_counts(normalize=True).items()},
        "weekend_share": float(weekend.mean()),
        "no_decision_day": int(d.isna().sum()),
        "decision_lag_days": {
            int(k): float(v) for k, v in lag.value_counts(normalize=True).items()
        },
        "same_day_decision_share": float((lag == 0).mean()),
    }


def headlines(news: pd.DataFrame) -> dict[str, Any]:
    h = news["headline"]
    length = h.str.len()
    checks = h.map(clean.check_headline)
    problems = pd.Series([c.problems for c in checks], index=news.index)
    counts = {
        k: int(problems.map(lambda p, k=k: k in p).sum())
        for k in ("too_short", "too_long", "listicle", "non_english")
    }
    q = length.quantile([0.01, 0.5, 0.99]).to_dict()
    examples = {
        k: news.loc[problems.map(lambda p, k=k: k in p), "headline"]
        .drop_duplicates()
        .head(3)
        .tolist()
        for k in counts
    }
    return {
        "chars_min": int(length.min()),
        "chars_p01": float(q[0.01]),
        "chars_median": float(q[0.5]),
        "chars_p99": float(q[0.99]),
        "chars_max": int(length.max()),
        "problem_counts": counts,
        "unusable": int((problems.map(len) > 0).sum()),
        "unusable_share": float((problems.map(len) > 0).mean()),
        "non_english_share": counts["non_english"] / len(news),
        "empty_summary_share": float((news["summary"].str.strip() == "").mean()),
        "nbsp_share": float(h.str.contains("\xa0").mean()),
        "examples": examples,
    }


def tickers_mapping(
    con: duckdb.DuckDBPyConnection, news: pd.DataFrame, universe: list[Any]
) -> dict[str, Any]:
    known = {c.ticker for c in universe}
    cfg = clean.config()["tickers"]
    per_article = news.groupby("id")["ticker"].nunique()
    roundup_ids = per_article[per_article >= int(cfg["roundup_min_tickers"])].index
    text = (news["headline"] + " " + news["summary"]).str.upper()
    mention = pd.Series(
        [
            bool(re.search(rf"\b{re.escape(t)}\b", s))
            for t, s in zip(news["ticker"], text, strict=True)
        ],
        index=news.index,
    )
    per_ticker = mention.groupby(news["ticker"]).mean()
    ciks = {c.ticker: set(c.ciks) for c in universe}
    fil = frame(con, "SELECT ticker, cik FROM filings WHERE available_at < ?", [dev_cut()])
    bad_cik = int(
        sum(1 for t, c in zip(fil["ticker"], fil["cik"], strict=True) if c not in ciks[t])
    )
    return {
        "news_tickers_outside_universe": sorted(set(news["ticker"]) - known),
        "universe_without_news": sorted(known - set(news["ticker"])),
        "roundup_min_tickers": int(cfg["roundup_min_tickers"]),
        "roundup_articles": int(len(roundup_ids)),
        "roundup_row_share": float(news["id"].isin(roundup_ids).mean()),
        "symbol_in_text_share": float(mention.mean()),
        "symbol_in_text_lowest": {t: float(v) for t, v in per_ticker.nsmallest(5).items()},
        "single_letter_tickers": sorted(t for t in known if len(t) == 1),
        "filings": len(fil),
        "filings_cik_mismatch": bad_cik,
    }


def filing_text(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    rows = con.execute(
        "SELECT accession_no, text_path FROM filings WHERE form = '8-K' AND available_at < ? "
        "AND text_path IS NOT NULL",
        [dev_cut()],
    ).fetchall()
    n = with_heading = tagged = leaked_after = inline_after = 0
    for _, path in rows:
        f = ROOT / path
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        n += 1
        with_heading += bool(_ITEM_HEADING.search(text))
        tagged += text.startswith("[")
        out = clean.strip_8k_item_headers(text)
        leaked_after += bool(_ITEM_HEADING.search(out))
        inline_after += bool(clean.ITEM_MENTION_RE.search(out))
    return {
        "texts": n,
        "with_item_heading": with_heading,
        "with_collector_tag": tagged,
        "headings_left_after_strip": leaked_after,
        "item_numbers_left_after_strip": inline_after,
    }


def audit(
    con: duckdb.DuckDBPyConnection, universe: list[Any] | None = None, benchmark: str = "SPY"
) -> dict[str, Any]:
    if universe is None:
        universe, benchmark = load_universe()
    tickers = [c.ticker for c in universe] + [benchmark]
    news = news_frame(con)
    return {
        "duplicates": duplicates(news),
        "missing_days": missing_days(con, news, tickers, benchmark),
        "utc_and_open": utc_and_open(con, news),
        "headlines": headlines(news),
        "tickers": tickers_mapping(con, news, universe),
        "filing_text": filing_text(con),
        "cut": dev_cut().isoformat(),
    }


# --- document ------------------------------------------------------------------------------------
def render(a: dict[str, Any], sha: str = "") -> str:
    d, m, u, h, t, f = (
        a["duplicates"],
        a["missing_days"],
        a["utc_and_open"],
        a["headlines"],
        a["tickers"],
        a["filing_text"],
    )
    ph = u["phase_share"]
    lag = u["decision_lag_days"]
    weekdays = ", ".join(m["weekdays_without_session"])
    below = (
        "\n".join(
            f"| {k} | {v['present']}/{v['expected']} | {', '.join(v['missing'])} |"
            for k, v in m["tickers_below_full"].items()
        )
        or "| none | | |"
    )
    ex = "\n".join(
        f"| {k} | {h['problem_counts'][k]:,} | {' · '.join(repr(x[:60]) for x in v) or '-'} |"
        for k, v in h["examples"].items()
    )
    lowest = ", ".join(f"{k} {_pct(v)}" for k, v in t["symbol_in_text_lowest"].items())
    return f"""# 04 — Data quality

Generated by `uv run python -m src.eda.quality` (git `{sha}`); numbers are recomputed on every run, the
rules below are enforced by `tests/nlp/test_clean.py`, `tests/test_clock.py`,
`tests/eda/test_quality_audit.py`. CRISP-DM phase 2–3 (data understanding / preparation).
Scope: development data only (warmup, train, validation); every query is cut at `{a["cut"]}`, the
first test cutoff, so the test period is not read (R3). Thresholds: `config/data_quality.yaml`.

## Summary of rules and handling

| Topic | Finding | Handling |
|---|---|---|
| Duplicates | {d["articles"]:,} distinct articles in {d["rows"]:,} rows; {_pct(d["multi_ticker_share"])} of articles are tagged with several tickers; {d["syndicated_copies"]:,} syndicated copies ({_pct(d["syndicated_share"])}) | the `(id, ticker)` key is kept (one article, many tickers is information, not an error); text-level duplicates are dropped by the relevance + dedup component with `clean.dedup_key` inside {d["window_hours"]} h |
| Missing trading days | prices below 100 % complete for {len(m["tickers_below_full"])} tickers; {len(m["weekdays_without_session"])} weekdays without a session (holidays) | exchange calendar = SPY sessions (D-29); gaps are warnings in `make collect-dev`, never filled |
| UTC and the ET open | all time columns are `TIMESTAMPTZ`; {_pct(ph.get("pre_open", 0))} of news is published before the open, {_pct(ph.get("session", 0))} during the session, {_pct(ph.get("after_close", 0))} after the close | store UTC, convert with `clock.to_utc`; the decision day is `clock.decision_day` (first session whose 09:00 ET cutoff is ≥ `available_at`) |
| Headline length | median {h["chars_median"]:.0f} chars, p99 {h["chars_p99"]:.0f}; {h["problem_counts"]["too_short"]} too short, {h["problem_counts"]["too_long"]} too long, {h["problem_counts"]["listicle"]} list items | `clean.check_headline` flags them; they are excluded from labelling sheets and sentiment evals, kept in the DB |
| Non-English | {h["problem_counts"]["non_english"]:,} headlines ({_pct(h["non_english_share"])}) | heuristic flag `non_english`; excluded from labelling and from the sentiment index |
| Ticker mapping | {len(t["news_tickers_outside_universe"])} tickers outside the universe; {_pct(t["roundup_row_share"])} of rows come from roundup articles tagged with ≥ {t["roundup_min_tickers"]} tickers; {t["filings_cik_mismatch"]} filings with a CIK that does not belong to their ticker | trust Finnhub's tag but down-weight roundups in relevance; filings are mapped by CIK (`former_ciks` for reorganisations) |
| 8-K item headers | {f["with_item_heading"]:,} of {f["texts"]:,} filing texts contain an `Item x.xx` heading | `clean.strip_8k_item_headers` before any text classification (the item number is the gold label) |

## 1. Duplicates

| Check | Value |
|---|---|
| news rows / distinct articles | {d["rows"]:,} / {d["articles"]:,} |
| duplicate `(id, ticker)` rows (primary key) | {d["pk_duplicates"]} |
| articles tagged with more than one ticker | {d["multi_ticker_articles"]:,} ({_pct(d["multi_ticker_share"])}) |
| rows sharing a normalised headline with another row of the same ticker | {d["same_ticker_same_headline_rows"]:,} |
| of those, repeats within {d["window_hours"]} h under a different id (syndicated copies) | {d["syndicated_copies"]:,} ({_pct(d["syndicated_share"])}) |

Why it matters: counting a syndicated story five times inflates a ticker's sentiment index and the
"news per day" attention measure. `dedup_key` lower-cases, folds no-break spaces (NFKC) and drops
punctuation, so "Apple: up 3%" and "apple up 3" collide while "up 4%" does not.

## 2. Missing trading days

Sessions per period (SPY as calendar): {", ".join(f"{k} {v}" for k, v in m["sessions"].items())}.
Weekdays inside the range without a session (must equal the NYSE holidays): {weekdays}.

| Ticker below 100 % | bars present | first missing dates |
|---|---|---|
{below}

Lowest completeness over the universe: {_pct(m["min_completeness"])}. Bars with non-positive prices or
high < low: {m["bad_bars"]}; zero-volume bars: {m["zero_volume_bars"]}.
News coverage in train + validation: {m["news_ticker_days_with_news"]:,} of {m["news_ticker_days"]:,}
ticker-days have at least one article ({_pct(m["news_ticker_day_coverage"])}); median articles on a
covered ticker-day: {m["news_per_ticker_day_median"]:.0f}. Least covered: {", ".join(f"{k} {_pct(v)}" for k, v in m["news_coverage_lowest"].items())}.
A ticker-day without news must stay a valid, neutral input for every strategy.

## 3. UTC normalisation and conversion to the ET open

- Columns that are not `TIMESTAMPTZ`: {u["non_timestamptz_columns"]} (must be 0). DuckDB hands values back in the
  *session* zone (this machine returned `Europe/Kyiv`), so every reader sets `TimeZone = 'UTC'`
  (`MarketView`, `src.eda.data.open_dev`) and every comparison goes through `clock.to_utc`.
  UTC offsets seen after normalisation: {", ".join(u["returned_utc_offsets"])}.
- News range: {u["published_min"]} .. {u["published_max"]}.
- Cutoff of day D = 09:00 America/New_York, i.e. 13:00 UTC in summer time and 14:00 UTC in winter
  time (`tests/test_clock.py` pins both and the weeks of the clock changes).
- Publication phase (New York time): pre-open {_pct(ph.get("pre_open", 0))} · session {_pct(ph.get("session", 0))} ·
  after close {_pct(ph.get("after_close", 0))}; published on a weekend: {_pct(u["weekend_share"])}.
- Decision day of a news item = first session whose cutoff is ≥ its `available_at`. Days between the
  New York publication date and that session: {", ".join(f"{k} d: {_pct(v)}" for k, v in sorted(lag.items()))}.
  Only {_pct(u["same_day_decision_share"])} of articles can be acted on at the open of the day they appear, because
  most are published after 09:00 ET; the rest wait for the next open. Items with no later session in the
  data: {u["no_decision_day"]:,} (the last development day).

## 4. Headline length

Characters: min {h["chars_min"]}, p1 {h["chars_p01"]:.0f}, median {h["chars_median"]:.0f}, p99 {h["chars_p99"]:.0f}, max {h["chars_max"]}.
Empty summary: {_pct(h["empty_summary_share"])}; headlines containing a no-break space: {_pct(h["nbsp_share"])}
(folded by `normalize_text`). Headlines unusable for labelling: {h["unusable"]:,} ({_pct(h["unusable_share"])}).

| Problem | Headlines | Examples |
|---|---|---|
{ex}

## 5. Non-English text

`clean.is_probably_english`: the share of letters in Latin script must be ≥ 90 %, and for Latin-script
languages at least two distinct foreign function words must outnumber the English ones. It is a
heuristic, not a language identifier; its false positives are listed in the examples above and
checked by hand before a labelling sheet is built.

## 6. Ticker mapping

- News tickers outside the universe: {", ".join(t["news_tickers_outside_universe"]) or "none"}; universe names without any news:
  {", ".join(t["universe_without_news"]) or "none"}.
- {t["roundup_articles"]:,} articles are tagged with ≥ {t["roundup_min_tickers"]} tickers ({_pct(t["roundup_row_share"])} of rows): market
  roundups and lists, not company news.
- The ticker symbol appears as a whole word in the headline or summary for {_pct(t["symbol_in_text_share"])} of rows
  (lowest: {lowest}). Single-letter symbols ({", ".join(t["single_letter_tickers"])}) match ordinary words, so the
  rate is an upper bound for them; company-name matching is a relevance-component task.
- Filings are mapped by CIK, never by text: {t["filings_cik_mismatch"]} of {t["filings"]:,} have a CIK outside their
  ticker's `cik` / `former_ciks`.

## 7. 8-K item headers

{f["texts"]:,} 8-K texts: {f["with_item_heading"]:,} contain at least one `Item x.xx` heading and {f["with_collector_tag"]:,} start
with the collector's `[EX-99.1] ...` tag. After `strip_8k_item_headers` {f["headings_left_after_strip"]} headings and
{f["item_numbers_left_after_strip"]} item numbers remain. Why: the item number is the gold label (D-10); a
classifier that reads "Item 5.02 Departure of Directors" is reading the answer.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    import subprocess

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    ).stdout.strip()
    con = open_dev()
    text = render(audit(con), sha)
    if not args.no_write:
        DOC.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
