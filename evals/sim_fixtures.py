"""Hand-worked simulator fixtures (docs/03_eval_plan.md, row "Simulator").

Each fixture is a tiny market (open/close per ticker per day), a scripted strategy and the final
equity worked out by hand in the comment. Strategies here read nothing from the database, so the
MarketView they receive never opens a connection. Shared by `evals/eval_simulator.py` and
`tests/sim/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.marketview import MarketView
from src.schemas import Signal
from src.sim.engine import SimConfig, SimResult, simulate
from src.sim.execution import ExecutionPrices
from src.strategies.base import make_signal
from src.strategies.buy_and_hold import BuyAndHold

D1, D2, D3 = date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3)  # validation days, Mon-Wed


class Scripted:
    """Score per (day, ticker) looked up from the view's cutoff date; 0 when unlisted."""

    def __init__(
        self, scores: dict[date, dict[str, float]], horizon_days: int | None = None
    ) -> None:
        self.name = "scripted"
        self.scores = scores
        self.horizon_days = horizon_days

    def signal(self, ticker: str, view: MarketView) -> Signal:
        day = view.as_of.date()  # 09:00 ET is 13:00/14:00 UTC: same calendar date
        return make_signal(self.name, ticker, view, self.scores.get(day, {}).get(ticker, 0.0))


def market(bars: dict[str, list[tuple[float, float]]], days: list[date]) -> ExecutionPrices:
    return ExecutionPrices(
        {(t, d): oc for t, rows in bars.items() for d, oc in zip(days, rows, strict=True)}, days
    )


@dataclass(frozen=True)
class Fixture:
    name: str
    strategy: object
    tickers: list[str]
    prices: ExecutionPrices
    cfg: SimConfig
    expected_final: float
    expected_trades: int

    def run(self) -> SimResult:
        return simulate(self.strategy, self.tickers, self.prices, self.cfg)  # type: ignore[arg-type]


def fixtures() -> list[Fixture]:
    return [
        # Buy-and-hold, A and B, cap 50 %, cost 10 bps. D1 open: 500 each -> qty_A = 500/(10*1.001),
        # qty_B = 500/(20*1.001). Last close: A 12, B 22, sold at (1-c):
        # 500/1.001 * (12/10 + 22/20) * 0.999 = 1147.7023...
        Fixture(
            "toy_buy_and_hold",
            BuyAndHold(),
            ["A", "B"],
            market(
                {"A": [(10, 11), (11, 12), (12, 12)], "B": [(20, 19), (19, 20), (20, 22)]},
                [D1, D2, D3],
            ),  # fmt: skip
            SimConfig(initial_capital=1000, max_position_pct=0.5, cost_per_side=0.001),
            500 / 1.001 * (12 / 10 + 22 / 20) * 0.999,
            4,
        ),
        # Flat price 10, buy at the D1 open and sell at the D1 close (h = 0), cost 5 bps:
        # qty = 1000/(10*1.0005); proceeds = 1000 * 0.9995/1.0005 = 999.0005; nothing else moves.
        Fixture(
            "round_trip_costs",
            Scripted({D1: {"A": 1.0}}, horizon_days=0),
            ["A"],
            market({"A": [(10, 10), (10, 10)]}, [D1, D2]),
            SimConfig(initial_capital=1000, max_position_pct=1.0, cost_per_side=0.0005),
            1000 * 0.9995 / 1.0005,
            2,
        ),
        # Score 1 on D1, 0 on D2: bought at the D1 open (10), sold at the D2 OPEN (15), not at
        # the D2 close (30); no cost. Cap 20 %: 200 invested -> 20 shares -> 300; cash 800.
        Fixture(
            "score_to_cash_at_open",
            Scripted({D1: {"A": 1.0}, D2: {"A": 0.0}}),
            ["A"],
            market({"A": [(10, 12), (15, 30), (30, 30)]}, [D1, D2, D3]),
            SimConfig(initial_capital=1000, max_position_pct=0.2, cost_per_side=0.0),
            800 + 20 * 15,
            2,
        ),
        # h = 1: bought at the D1 open (10), sold at the D2 close (14) although the score stays 1,
        # re-bought at the D3 open (20) and sold at the D3 close (end of split, 21); no cost.
        # 1000 -> 100 shares -> 1400 -> 70 shares -> 1470.
        Fixture(
            "horizon_exit_and_reentry",
            Scripted({D1: {"A": 1.0}, D2: {"A": 1.0}, D3: {"A": 1.0}}, horizon_days=1),
            ["A"],
            market({"A": [(10, 11), (12, 14), (20, 21)]}, [D1, D2, D3]),
            SimConfig(initial_capital=1000, max_position_pct=1.0, cost_per_side=0.0),
            1470.0,
            4,
        ),
    ]
