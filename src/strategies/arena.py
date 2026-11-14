"""`make arena` entry point (D-42). The arena itself lives in `src/sim/arena.py`, because it
replays execution prices and only `src/sim` may read open(D) (AGENTS.md R2)."""

from __future__ import annotations

from src.sim.arena import main

if __name__ == "__main__":
    raise SystemExit(main())
