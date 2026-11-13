"""Shared helper for routers whose service does not exist yet."""

from __future__ import annotations

from fastapi import HTTPException


def not_implemented(what: str) -> HTTPException:
    return HTTPException(status_code=501, detail=f"{what}: not implemented yet (design phase)")
