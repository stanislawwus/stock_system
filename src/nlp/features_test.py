"""`make features-test`: FinBERT inference on test-period news -> parquet (D-38, AGENTS.md R3).

The one module allowed to read test-period news text outside `make test-final`. It does exactly
what `make features` does for the dev splits (`features.pending` + `features.score_frame`, same
pinned FinBERT revision), but:

- it reads only the `news` and `sentiment` tables, through one short read-only connection that
  is closed before inference starts (R7: the single writer is never blocked for long, and this
  module never takes the write lock). No prices, returns, labels, MarketView or strategy;
- it writes `data/test_scores.parquet` (+ `.manifest.json`), never DuckDB;
- it is resumable: every chunk of `chunk_articles` articles is written atomically as one part
  file in `parts_dir`; a rerun skips (id, ticker) pairs already in a part;
- it reports counts, throughput and an ETA only, never score values or distributions, so nobody
  sees test-period outputs before the final run.

Window: (first test cutoff - `lookback_days_before_split`, cutoff of the last test day], minus
pairs already in the DuckDB `sentiment` table for the same method (the dev run covers those).
In the single final run MarketView adds these rows to `sentiment()` (src/splits.py
`final_scores_for`); the final-run context manager itself is unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src import db
from src.collectors.common import load_periods
from src.marketview.clock import cutoff_for
from src.nlp.features import SENTIMENT_COLUMNS, feature_tickers, pending, score_frame
from src.nlp.relevance import config as nlp_config
from src.nlp.sentiment import Backend, get_backend
from src.nlp.sentiment import cfg as sentiment_cfg
from src.splits import first_test_cutoff

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
META = "meta.json"


def cfg() -> dict[str, Any]:
    return nlp_config()["features_test"]


def default_paths() -> tuple[Path, Path]:
    c = cfg()
    return ROOT / c["scores_path"], ROOT / c["parts_dir"]


def window() -> tuple[datetime, datetime]:
    """(lo, hi] of available_at whose news feed a test-period decision."""
    pad = timedelta(days=int(nlp_config()["features"]["lookback_days_before_split"]))
    (test,) = load_periods(["test"])
    return first_test_cutoff() - pad, cutoff_for(test.end)


def model_meta(backend: Backend) -> dict[str, str]:
    """What produced the scores; a resume with a different model is refused."""
    s = sentiment_cfg()
    meta = {"method": backend.method}
    if backend.method == "finbert":
        meta |= {"model": s["finbert_model"], "revision": s["finbert_revision"]}
    return meta


def load_news(
    tickers: list[str], lo: datetime, hi: datetime, method: str, db_path: Path | None = None
) -> pd.DataFrame:
    """Test-window news without a DuckDB sentiment row. One read-only connection, closed here."""
    con = db.connect(db_path)  # read_only=True; never the writer lock (R7)
    try:
        con.execute("SET TimeZone = 'UTC'")
        rows = pending(con, tickers, lo, hi, method)
    finally:
        con.close()
    rows["available_at"] = pd.to_datetime(rows["available_at"], utc=True)
    rows["summary"] = rows["summary"].fillna("")
    return rows


# --------------------------------------------------------------------------- checkpoints


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _parts(parts_dir: Path) -> list[Path]:
    return sorted(parts_dir.glob("part-*.parquet"))


def done_pairs(parts_dir: Path) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for p in _parts(parts_dir):
        df = pd.read_parquet(p, columns=["source_ref", "ticker"])
        out.update(zip(df["source_ref"], df["ticker"], strict=True))
    return out


def check_meta(parts_dir: Path, meta: dict[str, str]) -> None:
    """First run writes meta.json; a resume must use the same model and revision."""
    path = parts_dir / META
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != meta:
            raise RuntimeError(
                f"{parts_dir} was produced by {old}, not {meta}; delete it to start over (D-38)"
            )
    else:
        parts_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _eta(done: int, total: int, seconds: float) -> str:
    if done == 0:
        return "unknown"
    left = seconds / done * (total - done)
    return str(timedelta(seconds=round(left)))


# --------------------------------------------------------------------------- service


def build(
    tickers: list[str] | None = None,
    backend: Backend | None = None,
    db_path: Path | None = None,
    scores_path: Path | None = None,
    parts_dir: Path | None = None,
    chunk: int | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Score every pending test-window article, checkpoint per chunk, write the parquet.

    Returns counts only (never scores). `progress` gets the same dict as the log line.
    """
    default_scores, default_parts = default_paths()
    scores_path = scores_path or default_scores
    parts_dir = parts_dir or default_parts
    chunk = chunk or int(cfg()["chunk_articles"])
    b = backend or get_backend(cfg()["backend"])
    meta = model_meta(b)
    check_meta(parts_dir, meta)

    lo, hi = window()
    tick = tickers or feature_tickers()
    rows = load_news(tick, lo, hi, b.method, db_path)
    already = done_pairs(parts_dir)
    if already:
        keys = list(zip(rows["id"], rows["ticker"], strict=True))
        rows = rows[[k not in already for k in keys]]
    ids = rows["id"].drop_duplicates().tolist()
    n_parts = len(_parts(parts_dir))
    log.info(
        "features-test: %d articles (%d rows) to score, %d rows already checkpointed",
        len(ids),
        len(rows),
        len(already),
    )

    t0 = time.monotonic()
    for start in range(0, len(ids), chunk):
        part_ids = set(ids[start : start + chunk])
        out = score_frame(rows[rows["id"].isin(part_ids)], b)
        _atomic_parquet(out, parts_dir / f"part-{n_parts:05d}.parquet")
        n_parts += 1
        done = min(start + chunk, len(ids))
        secs = time.monotonic() - t0
        state = {
            "articles_done": done,
            "articles_total": len(ids),
            "articles_per_s": round(done / secs, 2) if secs else None,
            "eta": _eta(done, len(ids), secs),
            "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        (parts_dir / "progress.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        log.info(
            "features-test: %d / %d articles, %s/s, ETA %s",
            done,
            len(ids),
            state["articles_per_s"],
            state["eta"],
        )
        if progress:
            progress(state)

    frames = [pd.read_parquet(p) for p in _parts(parts_dir)]
    allrows = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=SENTIMENT_COLUMNS).astype({"available_at": "datetime64[us, UTC]"})
    )
    allrows = allrows.sort_values(["available_at", "source_ref", "ticker"], ignore_index=True)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(allrows, scores_path)
    manifest = {
        **meta,
        "window": [lo.isoformat(), hi.isoformat()],
        "tickers": len(tick),
        "rows": len(allrows),
        "articles": int(allrows["source_ref"].nunique()),
        "complete": True,
        "git_sha": _git_sha(),
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    scores_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {
        "method": b.method,
        "new_rows": len(rows),
        "rows_total": len(allrows),
        "seconds": round(time.monotonic() - t0, 1),
        "path": str(scores_path),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tickers", default=None, help="comma list; default features.universe")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tickers = args.tickers.split(",") if args.tickers else None
    print(build(tickers))  # counts only (D-38)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
