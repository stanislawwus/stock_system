"""Random long-only controls with matched exposure and trade count (arena, D-42).

For a finished simulation the control keeps its *shape* and randomises its *timing and names*:
the same number of round trips, each with the same number of closes held, placed on a random
watchlist ticker at a random session of the same split. The control runs through the same
engine (same sizing, cap and costs), so the only thing it lacks is the strategy's choice of when
and what to buy. The percentile of the strategy's Sharpe among the controls answers "is this
better than random entries of the same size and frequency?".

Matching rules:
- a round trip keeps its length in sessions and its exit type: bought at the open of e, sold at
  the open of x (score exit, stop) or at the close of x (time exit, end of split). The control
  buys at the open of e' and sells at the open / close of e' + (x - e) the same way (close exits
  via the engine's per-entry `horizon_for`), so the engine's order of operations (exits at the
  open free cash before entries, close exits do not) and hence the sizing are reproduced.
- two modes (`random_schedule`): `names` keeps every entry session and randomises the ticker
  (same positions open on every day, so exposure matches almost exactly; isolates *what* to
  buy); `timing_and_names` also randomises the entry session (tests *when* as well).
- placement is uniform over every free (ticker, start) slot: a ticker is busy from its entry to
  its exit session, an entry needs an open price. A dense strategy can have trips that no longer
  fit (`timing_and_names`); they are dropped and the achieved trade count is reported.
- controls run without the risk manager (its vetoes would remove random trades and break the
  trade-count match); the strategy's own stops are already in its holding lengths.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

from src.marketview.clock import bar_close_for
from src.schemas import Signal
from src.sim.engine import SimConfig, SimResult, simulate
from src.sim.execution import ExecutionPrices
from src.sim.metrics import daily_returns, sharpe
from src.strategies.base import make_signal


@dataclass(frozen=True)
class RoundTrip:
    ticker: str
    entry: int  # session index of the buy (at the open)
    length: int  # exit session index - entry session index
    at_close: bool  # sold at the close of the exit session (else at its open)


def round_trips(res: SimResult) -> list[RoundTrip]:
    """Pair the buys and sells of a simulation into round trips (session indices)."""
    idx = {d: i for i, d in enumerate(res.days)}
    open_: dict[str, int] = {}
    out: list[RoundTrip] = []
    for tr in res.trades:
        i = idx[tr.day]
        if tr.side == "buy":
            open_[tr.ticker] = i
            continue
        e = open_.pop(tr.ticker)
        out.append(RoundTrip(tr.ticker, e, i - e, tr.available_at >= bar_close_for(tr.day)))
    return out


MODES = ("names", "timing_and_names")


def random_schedule(
    trips: list[RoundTrip],
    tickers: list[str],
    prices: ExecutionPrices,
    rng: np.random.Generator,
    mode: str = "timing_and_names",
) -> list[RoundTrip]:
    """Same lengths and exit types as `trips` on free slots drawn uniformly (see module doc)."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    n, k_all = len(prices.sessions), len(tickers)
    occ = np.zeros((k_all, n), dtype=bool)  # occ[k, s]: ticker k is held / exits at session s
    tradable = np.array(
        [[prices.open_price(t, d) is not None for d in prices.sessions] for t in tickers]
    )
    if mode == "names":
        order = sorted(trips, key=lambda tp: tp.entry)
    else:
        order = sorted(trips, key=lambda tp: -tp.length)  # long first: short ones fill the gaps
    out = []
    for tp in order:
        length = min(tp.length, n - 1)
        cs = np.concatenate([np.zeros((k_all, 1), int), np.cumsum(occ, axis=1)], axis=1)
        if mode == "names":
            starts = np.array([min(tp.entry, n - 1 - length)])
        else:
            starts = np.arange(n - length)
        busy = cs[:, starts + length + 1] - cs[:, starts]  # sessions e..e+length
        free = np.argwhere((busy == 0) & tradable[:, starts])
        if not len(free):
            continue  # no slot left: the trip is dropped (the achieved count is reported)
        k, j = free[int(rng.integers(len(free)))]
        e = int(starts[j])
        occ[k, e : e + length + 1] = True
        out.append(RoundTrip(tickers[int(k)], e, int(length), tp.at_close))
    return out


class _ClockView:
    """A view that only knows its cutoff: the control never reads data."""

    def __init__(self, as_of: datetime) -> None:
        self.as_of = as_of

    def __enter__(self) -> _ClockView:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class ScheduledStrategy:
    """Score 1 while a scheduled trip is held; 0 on the exit session of an open exit (cash at
    that open); close exits end through `horizon_for` (the engine's time exit)."""

    name = "random_control"
    horizon_days = None
    min_hold_days = 0

    def __init__(self, schedule: list[RoundTrip], sessions: list[date]) -> None:
        self.hold: set[tuple[str, date]] = set()
        self.horizons: dict[tuple[str, date], int] = {}
        for tp in schedule:
            last = tp.entry + tp.length if tp.at_close else tp.entry + tp.length - 1
            self.hold.update((tp.ticker, sessions[i]) for i in range(tp.entry, last + 1))
            if tp.at_close:
                self.horizons[(tp.ticker, sessions[tp.entry])] = tp.length

    def horizon_for(self, ticker: str, day: date) -> int | None:
        return self.horizons.get((ticker, day))

    def signal(self, ticker: str, view: _ClockView) -> Signal:
        on = (ticker, view.as_of.date()) in self.hold
        return make_signal(self.name, ticker, view, 1.0 if on else 0.0)  # type: ignore[arg-type]


def run_control(
    schedule: list[RoundTrip], tickers: list[str], prices: ExecutionPrices, cfg: SimConfig
) -> SimResult:
    strat = ScheduledStrategy(schedule, prices.sessions)
    # the entry threshold is the strategy's business; a control enters on every scheduled day
    plain = SimConfig(cfg.initial_capital, cfg.max_position_pct, cfg.cost_per_side, 0.0)
    return simulate(strat, tickers, prices, plain, _ClockView)  # type: ignore[arg-type]


def random_controls(
    res: SimResult,
    tickers: list[str],
    prices: ExecutionPrices,
    cfg: SimConfig,
    *,
    n: int,
    seed: int,
    days_per_year: int,
    mode: str = "timing_and_names",
) -> dict[str, float | int | str | None]:
    """Sharpe distribution of `n` matched random controls and the strategy's percentile in it."""
    trips = round_trips(res)
    ours = sharpe(daily_returns(res.equity, cfg.initial_capital), days_per_year)
    our_exposure = float(np.mean(res.exposure)) if res.exposure else 0.0
    if not trips:
        return {"n": 0, "mode": mode, "round_trips": 0, "sharpe": ours, "percentile": None}
    rng = np.random.default_rng(seed)
    sharpes, n_trips, exposure = [], [], []
    for _ in range(n):
        sim = run_control(random_schedule(trips, tickers, prices, rng, mode), tickers,
                          prices, cfg)  # fmt: skip
        s = sharpe(daily_returns(sim.equity, cfg.initial_capital), days_per_year)
        sharpes.append(0.0 if s is None else s)
        n_trips.append(len(sim.round_trip_pnls))
        exposure.append(float(np.mean(sim.exposure)))
    arr = np.asarray(sharpes)
    pct = None
    if ours is not None:
        pct = float(100 * (np.mean(arr < ours) + 0.5 * np.mean(arr == ours)))
    q05, q50, q95 = np.quantile(arr, [0.05, 0.5, 0.95])
    return {
        "n": n,
        "mode": mode,
        "seed": seed,
        "sharpe": ours,
        "percentile": pct,
        "random_sharpe_p05": float(q05),
        "random_sharpe_median": float(q50),
        "random_sharpe_p95": float(q95),
        "round_trips": len(trips),
        "random_round_trips_mean": float(np.mean(n_trips)),
        "mean_exposure": our_exposure,
        "random_mean_exposure": float(np.mean(exposure)),
    }
