"""Simulator accounting (src/sim/engine.py): toy example, costs, cash on score <= 0, caps."""

from __future__ import annotations

import pytest

from evals import eval_simulator
from evals.sim_fixtures import D1, D2, D3, Scripted, fixtures, market
from src.collectors.common import load_yaml
from src.sim.engine import SimConfig, simulate
from src.strategies.buy_and_hold import BuyAndHold

FX = {f.name: f for f in fixtures()}


def test_hand_computed_toy_buy_and_hold() -> None:
    """A: open 10 -> closes 11, 12, 12; B: open 20 -> closes 19, 20, 22. 1000 USD, cap 50 %,
    10 bps per side. D1 open: 500 each, qty_A = 500/(10*1.001), qty_B = 500/(20*1.001)."""
    res = FX["toy_buy_and_hold"].run()
    k = 500 / 1.001  # USD of stock bought per name, after the buy cost
    assert res.equity[0] == pytest.approx(k * (11 / 10 + 19 / 20), abs=1e-9)  # 1023.976...
    assert res.equity[1] == pytest.approx(k * (12 / 10 + 20 / 20), abs=1e-9)  # 1098.901...
    assert res.equity[2] == pytest.approx(k * (12 / 10 + 22 / 20) * 0.999, abs=1e-9)  # 1147.70
    buys = [t for t in res.trades if t.side == "buy"]
    assert [(t.ticker, t.day, t.price) for t in buys] == [("A", D1, 10), ("B", D1, 20)]
    assert buys[0].qty == pytest.approx(500 / (10 * 1.001))
    sells = [t for t in res.trades if t.side == "sell"]
    assert {(t.day, t.reason) for t in sells} == {(D3, "end_of_split")}
    # costs: buys 1000/1.001 * 0.001; sells k * (1.2 + 1.1) * 0.001
    assert res.costs_paid == pytest.approx(1000 / 1.001 * 0.001 + k * 2.3 * 0.001)


@pytest.mark.parametrize("name", sorted(FX))
def test_every_hand_fixture_matches_to_the_cent(name: str) -> None:
    fx = FX[name]
    res = fx.run()
    assert res.equity[-1] == pytest.approx(fx.expected_final, abs=0.005)
    assert len(res.trades) == fx.expected_trades


def test_round_trip_cost_is_exactly_two_sides() -> None:
    res = FX["round_trip_costs"].run()
    notional = 1000 / 1.0005  # stock value bought with 1000 USD at 5 bps
    assert res.costs_paid == pytest.approx(2 * notional * 0.0005)
    assert res.equity[-1] == pytest.approx(1000 - res.costs_paid)  # flat price: only costs lost
    assert sum(t.commission for t in res.trades) == pytest.approx(res.costs_paid)
    assert res.round_trip_pnls == [pytest.approx(-res.costs_paid)]
    assert res.traded_notional == pytest.approx(2 * notional)


def test_configured_cost_is_5_bps_per_side() -> None:
    cfg = SimConfig.from_risk(load_yaml("risk.yaml"))
    assert cfg.cost_per_side == pytest.approx(0.0005)
    assert 0 < cfg.max_position_pct <= 1


@pytest.mark.parametrize("score", [0.0, -0.4, -1.0])
def test_score_at_or_below_zero_means_cash(score: float) -> None:
    prices = market({"A": [(10, 11), (11, 12)]}, [D1, D2])
    cfg = SimConfig(initial_capital=1000, max_position_pct=1.0, cost_per_side=0.0005)
    res = simulate(Scripted({D1: {"A": score}, D2: {"A": score}}), ["A"], prices, cfg)
    assert res.trades == []
    assert res.equity == [1000, 1000]
    assert res.exposure == [0, 0]


def test_held_position_is_closed_at_the_open_when_the_score_drops() -> None:
    res = FX["score_to_cash_at_open"].run()
    sell = res.trades[-1]
    assert (sell.side, sell.day, sell.price, sell.reason) == ("sell", D2, 15, "score<=0")
    assert sell.available_at.hour == 13 and sell.available_at.minute == 30  # 09:30 EDT
    assert res.exposure[1] == 0


def test_position_cap_and_no_leverage() -> None:
    prices = market({t: [(10, 10), (10, 10)] for t in "ABCDE"}, [D1, D2])
    cfg = SimConfig(initial_capital=1000, max_position_pct=0.3, cost_per_side=0.0)
    res = simulate(Scripted({D1: {"A": 1.0}}), list("ABCDE"), prices, cfg)
    assert res.exposure[0] == pytest.approx(0.3)  # one name: capped, the rest stays cash
    res = simulate(Scripted({D1: dict.fromkeys("ABCDE", 1.0)}), list("ABCDE"), prices, cfg)
    assert res.exposure[0] == pytest.approx(1.0)  # five names: 1/5 each, below the cap
    buys = [t.qty * t.price for t in res.trades if t.side == "buy"]
    assert buys == [pytest.approx(200)] * 5 and sum(buys) <= 1000


def test_equal_weight_counts_held_positions_and_never_overspends() -> None:
    """D1: A alone gets the 60 % cap. D2: B and C join, N = 3 -> 1/3 of equity each, but only the
    remaining cash (40 %) is there: B gets 1/3, C the rest. Cash never goes negative."""
    prices = market({t: [(10, 10), (10, 10), (10, 10)] for t in "ABC"}, [D1, D2, D3])
    cfg = SimConfig(initial_capital=900, max_position_pct=0.6, cost_per_side=0.0)
    sched = {D1: {"A": 1.0}, D2: dict.fromkeys("ABC", 1.0), D3: dict.fromkeys("ABC", 1.0)}
    res = simulate(Scripted(sched), list("ABC"), prices, cfg)
    d2 = {t.ticker: t.qty * t.price for t in res.trades if t.day == D2}
    assert d2 == {"B": pytest.approx(300), "C": pytest.approx(60)}
    assert res.exposure[1] == pytest.approx(1.0)


def test_ticker_without_a_bar_is_not_traded_that_day() -> None:
    prices = market({"A": [(10, 10), (10, 12)]}, [D1, D2])
    prices = type(prices)({k: v for k, v in prices._bars.items() if k[1] != D1}, [D1, D2])
    cfg = SimConfig(initial_capital=1000, max_position_pct=1.0, cost_per_side=0.0)
    res = simulate(BuyAndHold(), ["A"], prices, cfg)
    assert [t.day for t in res.trades if t.side == "buy"] == [D2]


def test_simulator_eval_passes_every_fixture() -> None:
    result = eval_simulator.evaluate()
    assert result["value"] == 1.0, result["details"]
    assert result["n"] == len(FX)
