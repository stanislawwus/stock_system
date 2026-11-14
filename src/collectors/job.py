"""The API collect job: the same `run()` as `make collect-*`, behind two gates (R3, R7, D-46).

The API never writes unless the deployment opts in (env var named by `collect.enabled_env` set to
"1"), and the test period is never collectable through the API (`make collect-test` right before
`make test-final` only).
"""

from __future__ import annotations

import os
from typing import Any

from src.briefing import api_config
from src.collectors.run import run


class CollectForbidden(PermissionError):
    """The request breaks a gate: API collect disabled, or the test period was asked for."""


def check_collect_allowed(periods: list[str]) -> None:
    cfg = api_config()["collect"]
    if os.environ.get(cfg["enabled_env"]) != "1":
        raise CollectForbidden(
            "collection through the API is disabled; use `make collect-dev` "
            f"(or set {cfg['enabled_env']}=1 for the API process)"
        )
    bad = sorted(set(periods) & set(cfg["forbidden_periods"]))
    if bad:
        raise CollectForbidden(f"period(s) {bad} are not collectable through the API (R3, D-13)")


def collect_job(periods: list[str]) -> dict[str, Any]:
    check_collect_allowed(periods)
    summary = run(periods, list(api_config()["collect"]["sources"]))
    return {
        "periods": periods,
        "quality": summary.get("quality"),
        "tickers": summary.get("tickers"),
    }
