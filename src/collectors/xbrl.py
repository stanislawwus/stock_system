"""XBRL companyfacts -> facts (revenue, diluted EPS, net income) for each 10-Q / 10-K.

available_at = acceptance time of the filing that carried the fact, looked up by accession number
in `filings` (D-30). When the accession is unknown, filed date at 22:00 ET (the EDGAR window closes
then, so acceptance cannot be later). Only the filing's own reporting period is kept (the latest
period end in that filing for the concept); comparative prior-year values are not re-stamped.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from typing import Any

import duckdb
import httpx
import pandas as pd

from src.collectors.common import (
    Company,
    Period,
    PoliteClient,
    collectors_config,
    log,
    raw_dir,
    read_json,
    write_json,
)
from src.collectors.edgar import market_date, sec_client
from src.marketview.clock import MARKET_TZ

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
UNITS = {"revenue": ("USD", "USD"), "net_income": ("USD", "USD"),
         "eps_diluted": ("USD/shares", "USD/share")}  # fmt: skip
FACT_COLS = ["ticker", "name", "value", "unit", "period_end", "fiscal_period", "method",
             "source_ref", "grounding", "available_at"]  # fmt: skip


def fact_available_at(
    accn: str, filed: str, acceptance: dict[str, datetime], fallback_local: time
) -> datetime:
    if accn in acceptance:
        return acceptance[accn]
    d = date.fromisoformat(filed)
    return datetime.combine(d, fallback_local, tzinfo=MARKET_TZ).astimezone(UTC)


def _duration_ok(entry: dict[str, Any], cfg: dict[str, Any]) -> bool:
    if "start" not in entry:
        return False
    days = (date.fromisoformat(entry["end"]) - date.fromisoformat(entry["start"])).days
    q_lo, q_hi = cfg["quarter_days"]
    a_lo, a_hi = cfg["annual_days"]
    if entry.get("form") == "10-K":
        return a_lo <= days <= a_hi
    return q_lo <= days <= q_hi


def split_by_period(
    data: dict[str, Any],
    periods: list[Period],
    acceptance: dict[str, datetime],
) -> dict[str, dict[str, dict[str, list]]]:
    """Reduce raw companyfacts to the concepts we use, bucketed by the period of availability.

    Returns {period: {concept: {unit: [entries]}}}; entries outside all periods are dropped, so
    test-period values never reach the dev raw cache (D-13).
    """
    cfg = collectors_config()["xbrl"]
    fallback_local = time.fromisoformat(cfg["fallback_local_time"])
    wanted = {c for concepts in cfg["concepts"].values() for c in concepts}
    out: dict[str, dict[str, dict[str, list]]] = {p.name: {} for p in periods}
    gaap = data.get("facts", {}).get("us-gaap", {})
    for concept in wanted & set(gaap):
        for unit, entries in gaap[concept].get("units", {}).items():
            for e in entries:
                if e.get("form") not in cfg["forms"]:
                    continue
                avail = fact_available_at(e["accn"], e["filed"], acceptance, fallback_local)
                for p in periods:
                    if p.contains(market_date(avail)):
                        out[p.name].setdefault(concept, {}).setdefault(unit, []).append(e)
    return out


def fact_rows(
    company: Company, reduced: dict[str, dict[str, list]], acceptance: dict[str, datetime]
) -> list[dict[str, Any]]:
    cfg = collectors_config()["xbrl"]
    fallback_local = time.fromisoformat(cfg["fallback_local_time"])
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for name, concepts in cfg["concepts"].items():
        src_unit, unit = UNITS[name]
        for concept in concepts:  # priority order: first concept present wins per (accn, end)
            raw = reduced.get(concept, {}).get(src_unit, [])
            entries = [e for e in raw if _duration_ok(e, cfg)]
            latest_end: dict[str, str] = {}
            for e in entries:
                latest_end[e["accn"]] = max(latest_end.get(e["accn"], ""), e["end"])
            for e in entries:
                if e["end"] != latest_end[e["accn"]]:
                    continue  # comparative period, not this filing's own
                key = (name, e["accn"], e["end"])
                if key in rows:
                    continue
                rows[key] = {
                    "ticker": company.ticker,
                    "name": name,
                    "value": float(e["val"]),
                    "unit": unit,
                    "period_end": date.fromisoformat(e["end"]),
                    "fiscal_period": f"{e.get('fp', '?')}-{e.get('fy', '?')}",
                    "method": "xbrl",
                    "source_ref": e["accn"],
                    "grounding": None,
                    "available_at": fact_available_at(
                        e["accn"], e["filed"], acceptance, fallback_local
                    ),
                }
    return list(rows.values())


def facts_by_period(
    client: PoliteClient | None,
    cik: str,
    periods: list[Period],
    acceptance: dict[str, datetime],
) -> dict[str, dict[str, dict[str, list]]]:
    paths = {p.name: raw_dir() / "edgar" / p.name / "companyfacts" / f"{cik}.json"
             for p in periods}  # fmt: skip
    cached = {name: read_json(path) for name, path in paths.items()}
    if all(v is not None for v in cached.values()):
        return cached  # type: ignore[return-value]
    if client is None:
        raise RuntimeError(f"no cache for CIK {cik} and no client")
    try:
        data = client.get(COMPANYFACTS_URL.format(cik=cik)).json()
    except httpx.HTTPStatusError as exc:  # e.g. 404: the company files no XBRL
        log.warning("companyfacts CIK %s: %s", cik, exc)
        data = {}
    reduced = split_by_period(data, periods, acceptance)
    for name, path in paths.items():
        write_json(path, reduced[name])
    return reduced


def insert_facts(con: duckdb.DuckDBPyConnection, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)[FACT_COLS]
    before = con.execute("SELECT count(*) FROM facts").fetchall()[0][0]
    con.register("facts_df", df)
    con.execute(f"INSERT OR IGNORE INTO facts SELECT {', '.join(FACT_COLS)} FROM facts_df")
    con.unregister("facts_df")
    return con.execute("SELECT count(*) FROM facts").fetchall()[0][0] - before


def acceptance_map(con: duckdb.DuckDBPyConnection, ticker: str) -> dict[str, datetime]:
    rows = con.execute(
        "SELECT accession_no, accepted_at FROM filings "
        "WHERE ticker = ? AND form IN ('10-Q', '10-K')",
        [ticker],
    ).fetchall()
    return {acc: ts.astimezone(UTC) for acc, ts in rows}


def collect_facts(
    con: duckdb.DuckDBPyConnection, companies: list[Company], periods: list[Period]
) -> dict[str, int]:
    client = sec_client()
    stats = {"facts": 0, "fallback_time": 0}
    try:
        for company in companies:
            acceptance = acceptance_map(con, company.ticker)
            for cik in company.ciks:
                reduced = facts_by_period(client, cik, periods, acceptance)
                for p in periods:
                    rows = fact_rows(company, reduced[p.name], acceptance)
                    stats["fallback_time"] += sum(r["source_ref"] not in acceptance for r in rows)
                    stats["facts"] += insert_facts(con, rows)
    finally:
        client.close()
    return stats
