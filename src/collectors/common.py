"""Shared collector plumbing: config, periods, universe, raw cache, polite HTTP (R2, R9, R15)."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv

from src.ratelimit import RateLimiter

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
PERIOD_NAMES = ("warmup", "train", "validation", "test")

log = logging.getLogger("src.collectors")


def load_yaml(name: str) -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def collectors_config() -> dict[str, Any]:
    return load_yaml("collectors.yaml")


def raw_dir() -> Path:
    return ROOT / collectors_config()["raw_dir"]


@dataclass(frozen=True)
class Period:
    name: str
    start: date
    end: date

    def contains(self, d: date) -> bool:
        return self.start <= d <= self.end


def load_periods(names: Iterable[str]) -> list[Period]:
    splits = load_yaml("splits.yaml")
    out = []
    for n in names:
        if n not in PERIOD_NAMES:
            raise ValueError(f"unknown period {n!r}; expected one of {PERIOD_NAMES}")
        out.append(Period(n, splits[n]["start"], splits[n]["end"]))
    return sorted(out, key=lambda p: p.start)


def test_period() -> Period:
    return load_periods(["test"])[0]


@dataclass(frozen=True)
class Company:
    ticker: str
    sector: str
    cik: str
    former_ciks: tuple[str, ...] = ()

    @property
    def ciks(self) -> tuple[str, ...]:
        """Current filer id first, then earlier ones (e.g. before a holding-company merger)."""
        return (self.cik, *self.former_ciks)


def load_universe() -> tuple[list[Company], str]:
    cfg = load_yaml("universe.yaml")
    companies = [
        Company(**{**s, "former_ciks": tuple(s.get("former_ciks", ()))}) for s in cfg["stocks"]
    ]
    return companies, cfg["benchmark"]


def watchlist_tickers() -> list[str]:
    return [s["ticker"] for s in load_yaml("watchlist.yaml")["stocks"]]


def env(name: str, *, required: bool = True) -> str:
    """Read a setting from the environment or .env (R9). Never log the value."""
    load_dotenv(ROOT / ".env")
    value = os.environ.get(name, "").strip().strip('"')
    if required and not value:
        raise RuntimeError(f"{name} is not set; add it to .env (see .env.example)")
    return value


def utc_now() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- raw cache


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- polite HTTP


class PoliteClient:
    """httpx client with a rate limiter, retries with backoff on 429/5xx/network errors."""

    def __init__(
        self,
        limiter: RateLimiter,
        *,
        headers: dict[str, str] | None = None,
        max_retries: int = 4,
        backoff: float = 2.0,
        timeout: float = 30.0,
    ) -> None:
        self.limiter = limiter
        self.max_retries = max_retries
        self.backoff = backoff
        self.client = httpx.Client(headers=headers or {}, timeout=timeout, follow_redirects=True)
        self.calls = 0

    def get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self.limiter.wait()
            self.calls += 1
            try:
                resp = self.client.get(url, params=params)
            except httpx.HTTPError as exc:
                last = exc
            else:
                if resp.status_code == 429 or resp.status_code >= 500:
                    last = httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}", request=resp.request, response=resp
                    )
                elif resp.status_code >= 400:
                    # own message: httpx's would include the query string (Finnhub token, R9)
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code} for {_redact(url)}",
                        request=resp.request,
                        response=resp,
                    )
                else:
                    return resp
            wait = self.backoff * 2 ** (attempt - 1)
            log.warning("GET %s failed (%s), retry %d in %.0fs", _redact(url), last, attempt, wait)
            time.sleep(wait)
        assert last is not None
        raise last

    def close(self) -> None:
        self.client.close()


def _redact(url: str) -> str:
    return url.split("token=")[0]
