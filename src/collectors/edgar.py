"""SEC EDGAR filings: submissions JSON -> 8-K / 10-Q / 10-K / Form 4 headers, 8-K item numbers,
EX-99.1 (press release) text. available_at = acceptanceDateTime (D-24; verified UTC, D-29).

Raw caches hold only rows inside the requested period (D-13): the submissions JSON is filtered
before it is written to `data/raw/edgar/<period>/submissions/<cik>.json`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src.collectors.common import (
    ROOT,
    Company,
    Period,
    PoliteClient,
    collectors_config,
    env,
    log,
    raw_dir,
    read_json,
    write_json,
    write_text,
)
from src.marketview.clock import MARKET_TZ
from src.ratelimit import RateLimiter
from src.textutil import strip_html

SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
COLUMNS = ("accessionNumber", "filingDate", "acceptanceDateTime", "form", "items",
           "primaryDocument", "reportDate")  # fmt: skip


def sec_client() -> PoliteClient:
    cfg = collectors_config()["edgar"]
    ua = env("SEC_USER_AGENT")
    if "@" not in ua:
        raise RuntimeError("SEC_USER_AGENT must contain a contact e-mail: 'AppName name@domain'")
    return PoliteClient(
        RateLimiter(int(cfg["requests_per_second"]), 1.0),
        headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"},
        max_retries=int(cfg["max_retries"]),
        backoff=float(cfg["retry_backoff_seconds"]),
    )


def parse_acceptance(value: str) -> datetime:
    """'2026-11-09T21:30:40.000Z' -> aware UTC datetime. The EDGAR index page shows the same
    moment in Eastern time (16:30:40), so the trailing Z is genuine (checked 2026-11-13, D-29)."""
    ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        raise ValueError(f"acceptanceDateTime without timezone: {value!r}")
    return ts.astimezone(UTC)


def market_date(ts: datetime) -> date:
    """Calendar date in America/New_York: the date a period boundary refers to."""
    return ts.astimezone(MARKET_TZ).date()


def filter_columns(block: dict[str, list], forms: Iterable[str], period: Period) -> dict[str, list]:
    """Keep rows of a column-oriented submissions block with a wanted form inside the period."""
    forms = set(forms)
    out: dict[str, list] = {c: [] for c in COLUMNS}
    for i, form in enumerate(block["form"]):
        if form not in forms:
            continue
        accepted = parse_acceptance(block["acceptanceDateTime"][i])
        if not period.contains(market_date(accepted)):
            continue
        for c in COLUMNS:
            out[c].append(block.get(c, [""] * len(block["form"]))[i])
    return out


def filing_rows(block: dict[str, list], company: Company, cik: str) -> list[dict[str, Any]]:
    rows = []
    cik_int = int(cik)
    for i, acc in enumerate(block["accessionNumber"]):
        accepted = parse_acceptance(block["acceptanceDateTime"][i])
        items = [s.strip() for s in str(block["items"][i] or "").split(",") if s.strip()]
        rows.append(
            {
                "accession_no": acc,
                "ticker": company.ticker,
                "cik": cik,
                "form": block["form"][i],
                "items": items,
                "accepted_at": accepted,
                "filed_date": date.fromisoformat(block["filingDate"][i]),
                "url": f"{ARCHIVES}/{cik_int}/{acc.replace('-', '')}/{acc}-index.htm",
                "text_path": None,
                "available_at": accepted,
                "_primary": block["primaryDocument"][i],
            }
        )
    return rows


def _submissions_blocks(client: PoliteClient, cik: str, earliest: str) -> list[dict]:
    first = client.get(SUBMISSIONS_URL.format(name=f"CIK{cik}.json")).json()
    blocks = [first["filings"]["recent"]]
    for f in first["filings"].get("files", []):
        if f["filingTo"] >= earliest:
            blocks.append(client.get(SUBMISSIONS_URL.format(name=f["name"])).json())
    return blocks


def submissions_by_period(
    client: PoliteClient | None, cik: str, periods: list[Period]
) -> dict[str, dict[str, list]]:
    """Filtered submissions per period, from the raw cache or one download for all periods."""
    cfg = collectors_config()["edgar"]
    paths = {p.name: raw_dir() / "edgar" / p.name / "submissions" / f"{cik}.json"
             for p in periods}  # fmt: skip
    cached = {name: read_json(path) for name, path in paths.items()}
    if all(v is not None for v in cached.values()):
        return cached  # type: ignore[return-value]
    if client is None:
        raise RuntimeError(f"no cache for CIK {cik} and no client")
    blocks = _submissions_blocks(client, cik, min(p.start for p in periods).isoformat())
    for p in periods:
        merged: dict[str, list] = {c: [] for c in COLUMNS}
        for b in blocks:
            part = filter_columns(b, cfg["forms"], p)
            for c in COLUMNS:
                merged[c].extend(part[c])
        write_json(paths[p.name], merged)
        cached[p.name] = merged
    return cached  # type: ignore[return-value]


# --------------------------------------------------------------------------- exhibit text

_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.IGNORECASE | re.DOTALL)
_HREF_RE = re.compile(r'href="([^"]+)"', re.IGNORECASE)


def parse_index_documents(index_html: str) -> list[dict[str, str]]:
    """Rows of the 'Document Format Files' table: description, href, type."""
    docs = []
    for row in _ROW_RE.findall(index_html):
        cells = _TD_RE.findall(row)
        if len(cells) < 4:
            continue
        href = _HREF_RE.search(cells[2])
        if not href:
            continue
        link = href.group(1).replace("/ix?doc=", "")
        docs.append(
            {
                "description": strip_html(cells[1]),
                "href": link,
                "type": strip_html(cells[3]).upper(),
            }
        )
    return docs


def choose_document(docs: list[dict[str, str]], form: str, exhibit_types: list[str]) -> dict | None:
    for wanted in exhibit_types:
        for d in docs:
            if d["type"] == wanted and d["href"].lower().endswith((".htm", ".html", ".txt")):
                return d
    for d in docs:
        if d["type"] == form:
            return d
    return None


def fetch_text(client: PoliteClient, row: dict[str, Any], period: Period) -> str | None:
    """Download the press-release exhibit (else the primary document); return the text path."""
    cfg = collectors_config()["edgar"]
    acc = row["accession_no"]
    base = raw_dir() / "edgar" / period.name
    text_path = base / "text" / f"{acc}.txt"
    if text_path.exists():
        return text_path.relative_to(ROOT).as_posix()
    index_path = base / "index" / f"{acc}.htm"
    if index_path.exists():
        index_html = index_path.read_text(encoding="utf-8")
    else:
        index_html = client.get(row["url"]).text
        write_text(index_path, index_html)
    doc = choose_document(parse_index_documents(index_html), row["form"], cfg["exhibit_types"])
    if doc is None:
        log.warning("no document found in %s", row["url"])
        return None
    url = "https://www.sec.gov" + doc["href"] if doc["href"].startswith("/") else doc["href"]
    raw = client.get(url).text
    write_text(base / "docs" / f"{acc}__{Path(doc['href']).name}", raw)
    text = strip_html(raw, keep_lines=True)[: int(cfg["max_text_chars"])]
    write_text(text_path, f"[{doc['type']}] {doc['description']}\n\n{text}")
    return text_path.relative_to(ROOT).as_posix()


# --------------------------------------------------------------------------- collect

FILING_COLS = ["accession_no", "ticker", "cik", "form", "items", "accepted_at", "filed_date", "url",
               "text_path", "available_at"]  # fmt: skip


def insert_filings(con: duckdb.DuckDBPyConnection, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    df = pd.DataFrame([{k: r[k] for k in FILING_COLS} for r in rows])
    before = con.execute("SELECT count(*) FROM filings").fetchall()[0][0]
    con.register("filings_df", df)
    con.execute(f"INSERT OR IGNORE INTO filings SELECT {', '.join(FILING_COLS)} FROM filings_df")
    con.unregister("filings_df")
    return con.execute("SELECT count(*) FROM filings").fetchall()[0][0] - before


def _with_texts(
    con: duckdb.DuckDBPyConnection,
    client: PoliteClient,
    rows: list[dict[str, Any]],
    period: Period,
    cfg: dict[str, Any],
    stats: dict[str, int],
) -> int:
    for r in rows:
        if r["form"] not in cfg["text_forms"]:
            continue
        try:
            r["text_path"] = fetch_text(client, r, period)
        except Exception as exc:  # one bad document is a gap, not a crash
            log.warning("text for %s failed: %s", r["accession_no"], exc)
        stats["texts" if r["text_path"] else "text_missing"] += 1
    return insert_filings(con, rows)


def collect_filings(
    con: duckdb.DuckDBPyConnection, companies: list[Company], periods: list[Period]
) -> dict[str, int]:
    cfg = collectors_config()["edgar"]
    client = sec_client()
    stats = {"filings": 0, "texts": 0, "text_missing": 0}
    try:
        for n, company in enumerate(companies, 1):
            for cik in company.ciks:
                by_period = submissions_by_period(client, cik, periods)
                for p in periods:
                    rows = filing_rows(by_period[p.name], company, cik)
                    stats["filings"] += _with_texts(con, client, rows, p, cfg, stats)
            log.info("edgar %d/%d %s: %d filings so far", n, len(companies), company.ticker,
                     stats["filings"])  # fmt: skip
    finally:
        client.close()
    return stats
