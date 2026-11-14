"""Write the app's OpenAPI document to docs/openapi.json (skill new-endpoint, step 7)."""

from __future__ import annotations

import json
from pathlib import Path

from src.api.main import app

OUT = Path(__file__).resolve().parents[2] / "docs" / "openapi.json"


def render() -> str:
    return json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    OUT.write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
