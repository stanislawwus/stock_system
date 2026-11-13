"""Ticker -> CIK map from SEC company_tickers.json; verifies (or rewrites) config/universe.yaml.

uv run python -m src.collectors.tickers           # check universe CIKs against SEC
uv run python -m src.collectors.tickers --write   # rewrite the cik fields from SEC
"""

from __future__ import annotations

import argparse
import re
import sys

from src.collectors.common import CONFIG_DIR, raw_dir, read_json, write_json
from src.collectors.edgar import sec_client

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def parse_company_tickers(data: dict) -> dict[str, dict[str, str]]:
    """{'0': {'cik_str': 320193, 'ticker': 'AAPL', 'title': 'Apple Inc.'}, ...} -> by ticker."""
    out: dict[str, dict[str, str]] = {}
    for row in data.values():
        t = str(row["ticker"]).upper()
        out.setdefault(t, {"cik": f"{int(row['cik_str']):010d}", "title": str(row["title"])})
    return out


def ticker_cik_map(refresh: bool = False) -> dict[str, dict[str, str]]:
    path = raw_dir() / "edgar" / "company_tickers.json"
    data = None if refresh else read_json(path)
    if data is None:
        client = sec_client()
        try:
            data = client.get(COMPANY_TICKERS_URL).json()
        finally:
            client.close()
        write_json(path, data)
    return parse_company_tickers(data)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="rewrite cik fields in universe.yaml")
    ap.add_argument("--refresh", action="store_true", help="re-download company_tickers.json")
    args = ap.parse_args(argv)
    mapping = ticker_cik_map(refresh=args.refresh)
    path = CONFIG_DIR / "universe.yaml"
    text = path.read_text(encoding="utf-8")
    bad = 0
    for m in re.finditer(r"\{ticker: (\w+), sector: (\w+), cik: '(\d+)'", text):
        ticker, _, cik = m.groups()
        sec = mapping.get(ticker)
        if sec is None:
            print(f"MISSING {ticker}: not in company_tickers.json")
            bad += 1
        elif sec["cik"] != cik:
            print(f"MISMATCH {ticker}: config {cik} vs SEC {sec['cik']} ({sec['title']})")
            bad += 1
            if args.write:
                text = text.replace(m.group(0), m.group(0).replace(cik, sec["cik"]))
    if args.write:
        path.write_text(text, encoding="utf-8", newline="\n")
    print(f"checked universe: {bad} problem(s){' fixed' if args.write and bad else ''}")
    return 1 if bad and not args.write else 0


if __name__ == "__main__":
    sys.exit(main())
