"""Build the labelling sheet for the news sentiment + event gold set (label-helper skill).

    uv run python -m evals.make_news_sheet

Writes `evals/labels/news_100.draft.csv` (the sheet a human fills: `sentiment` and `events` are
BLANK) and `evals/labels/news_100.strata.csv` (how each row was drawn; kept apart so the annotator
does not see the 8-K item). Nothing is pre-filled, not even by a model: a model's suggestion in the
sheet would make the later eval circular (R4).

Population: English headlines of train + validation that pass `clean.check_headline`, are not
roundup articles (tagged with >= `tickers.roundup_min_tickers` tickers) and are unique by
`dedup_key`. Strata: coincident 8-K item group (same ticker, accepted within +/- the configured
window; "coincident" says nothing about cause) x year x ticker. Within an item group the draw
round-robins over (year, ticker) cells, alternating years, so no ticker or year dominates.
Seeded: the same database gives the same sheet. Test-period rows are never read (D-33).
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src.collectors.common import ROOT, load_periods
from src.eda.data import dev_cut, frame, open_dev
from src.nlp import clean

OUT = ROOT / "evals" / "labels"
EVENT_LABELS = "1.01 2.01 2.02 2.05 3.02 5.02 5.07 7.01 8.01 9.01 other_item none"
HEADER = [
    "# gold_set: news_100 (labelling sheet, NOT frozen until the human columns are filled)",
    "# created: 2026-11-14  by: auto:evals/make_news_sheet.py (sampling only)  "
    "llm_suggestions: no  kappa_vs_llm: n/a",
    "# source: Finnhub company news, train + validation, sampled stratified by coincident 8-K item "
    "group, year, ticker (see news_100.strata.csv)",
    "# sentiment: one of pos / neg / neu, from the view of a holder of the TAGGED ticker, headline "
    "only. No clear direction, or mixed = neu.",
    f"# events: one or more of [{EVENT_LABELS}] separated by a vertical bar; 'none' when no "
    "corporate event is reported (opinion, price move, listicle). Multi-label: a results + "
    "CEO-change headline is 2.02|5.02.",
]
COLUMNS = [
    "id",
    "ticker",
    "published_at",
    "year",
    "headline",
    "url",
    "sentiment",
    "events",
    "notes",
]


def candidates(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Usable, unique, non-roundup headlines of train + validation."""
    cfg = clean.config()
    periods = load_periods(("train", "validation"))
    lo, hi = periods[0].start, periods[-1].end
    news = frame(
        con,
        "SELECT id, ticker, published_at, headline, url FROM news WHERE available_at < ?",
        [dev_cut()],
    )
    ts = pd.to_datetime(news["published_at"], utc=True)
    local = ts.dt.tz_convert("America/New_York").dt.date
    news = news.assign(published_at=ts, year=ts.dt.year)
    news = news[(local >= lo) & (local <= hi)]
    n_tickers = news.groupby("id")["ticker"].transform("nunique")
    news = news[n_tickers < int(cfg["tickers"]["roundup_min_tickers"])]
    news = news[news["headline"].map(lambda h: clean.check_headline(h).ok)]
    news = news.assign(key=news["headline"].map(clean.dedup_key))
    news = news.sort_values(["published_at", "id", "ticker"])
    return news.drop_duplicates("key").reset_index(drop=True)


def stratify(con: duckdb.DuckDBPyConnection, cand: pd.DataFrame) -> pd.DataFrame:
    """Add `stratum` (coincident 8-K item group or "none") and `accession`."""
    cfg = clean.config()["label_sheet"]
    window = pd.Timedelta(hours=int(cfg["coincident_window_hours"]))
    fil = frame(
        con,
        "SELECT accession_no, ticker, items, accepted_at FROM filings "
        "WHERE form = '8-K' AND available_at < ?",
        [dev_cut()],
    )
    fil["accepted_at"] = pd.to_datetime(fil["accepted_at"], utc=True)
    order = [*cfg["groups"], "other"]

    def group(items: Any) -> str:
        for name, members in cfg["groups"].items():
            if any(i in members for i in items):
                return name
        return "other"

    fil["group"] = fil["items"].map(group)
    fil["rank"] = fil["group"].map(order.index)
    m = cand[["id", "ticker", "published_at"]].merge(
        fil[["ticker", "accepted_at", "accession_no", "group", "rank"]], on="ticker"
    )
    m = m[(m["accepted_at"] - m["published_at"]).abs() <= window]
    best = m.sort_values(["rank", "accession_no"]).drop_duplicates(["id", "ticker"])
    out = cand.merge(
        best[["id", "ticker", "group", "accession_no"]], on=["id", "ticker"], how="left"
    )
    out["stratum"] = out["group"].fillna("none")
    out["accession"] = out["accession_no"].fillna("")
    return out.drop(columns=["group", "accession_no"])


def _take(pool: pd.DataFrame, quota: int, rng: random.Random) -> list[int]:
    """Round-robin over (year, ticker) cells, alternating years; returns row labels of `pool`."""
    cells: dict[tuple[int, str], list[int]] = {}
    for idx, y, t in zip(pool.index, pool["year"], pool["ticker"], strict=True):
        cells.setdefault((int(y), t), []).append(idx)
    for rows in cells.values():
        rng.shuffle(rows)
    by_year: dict[int, list[tuple[int, str]]] = {}
    for cell in sorted(cells):
        by_year.setdefault(cell[0], []).append(cell)
    for lst in by_year.values():
        rng.shuffle(lst)
    chosen: list[int] = []
    while len(chosen) < quota and any(cells.values()):
        for year in sorted(by_year):
            lst = by_year[year]
            for _ in range(len(lst)):
                cell = lst.pop(0)
                lst.append(cell)
                if cells[cell]:
                    chosen.append(cells[cell].pop())
                    break
            if len(chosen) >= quota:
                break
    return chosen


def draw(strat: pd.DataFrame, quotas: dict[str, int], seed: int) -> pd.DataFrame:
    """Per stratum quota; what a thin stratum cannot supply is added to `none` (listed last)."""
    rng = random.Random(seed)
    picks: list[pd.DataFrame] = []
    shortfall = 0
    for name, quota in quotas.items():
        pool = strat[strat["stratum"] == name]
        want = quota + (shortfall if name == "none" else 0)
        chosen = _take(pool, want, rng)
        if name != "none":
            shortfall += quota - len(chosen)
        picks.append(pool.loc[chosen])
    return pd.concat(picks)


def build(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = clean.config()["label_sheet"]
    strat = stratify(con, candidates(con))
    drawn = draw(strat, {k: int(v) for k, v in cfg["quotas"].items()}, int(cfg["seed"]))
    drawn = drawn.sort_values(["published_at", "id", "ticker"]).reset_index(drop=True)
    drawn["published_at"] = drawn["published_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    drawn["headline"] = drawn["headline"].map(clean.normalize_text)
    for col in ("sentiment", "events", "notes"):
        drawn[col] = ""  # BLANK on purpose
    return drawn[COLUMNS], drawn[["id", "ticker", "stratum", "accession"]]


class FrozenGoldSetError(RuntimeError):
    """The labelled sheet was frozen (label-helper step 7); its strata must not be redrawn."""


def write(sheet: pd.DataFrame, strata: pd.DataFrame, out_dir: Path = OUT) -> None:
    name = clean.config()["label_sheet"]["name"]
    if (out_dir / f"{name}.csv").exists():
        raise FrozenGoldSetError(
            f"{name}.csv is a frozen gold set (AGENTS.md R4); a new sample goes to "
            f"{name}.v2 through the label-helper skill"
        )
    with open(out_dir / f"{name}.draft.csv", "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(HEADER) + "\n")
        sheet.to_csv(f, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    strata.to_csv(out_dir / f"{name}.strata.csv", index=False, lineterminator="\n")


def main() -> int:
    sheet, strata = build(open_dev())
    write(sheet, strata)
    name = clean.config()["label_sheet"]["name"]
    print(f"wrote {len(sheet)} rows to evals/labels/{name}.draft.csv")
    print(strata["stratum"].value_counts().to_string())
    print(sheet["ticker"].nunique(), "tickers;", sheet["year"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
