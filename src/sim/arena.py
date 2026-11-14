"""Strategy arena on validation (`make arena`, D-42). Never the test period (R3).

Runs every strategy of `arena.strategies` through the simulator on the same days and writes
`evals/results/arena.json` (AGENTS.md section 5 record) and `docs/05_arena.md`:

1. Leaderboard: Sharpe after costs with a 95 % moving-block bootstrap CI, return, drawdown,
   trades, exposure; plus the Sharpe on val_b alone, because `impact` and `aggregate_equal`
   chose their calibrator and hold/threshold on val_a (D-40): their full-validation figure is
   partly in-sample, the val_b one is not.
2. Correlation matrix of daily returns (Pearson).
3. Drop-one ablation of `arena.ablation_of`: the component scores are recorded once during the
   full run and replayed without one component, same hold/threshold, same risk manager.
4. Random controls (`src/sim/controls.py`): `random_controls.n` strategies with the same number
   of round trips and holding lengths at random names/days; the percentile of our Sharpe.
5. Deflated Sharpe (Bailey & Lopez de Prado 2014) with N = every configuration in the trials log
   (`arena.trials_log`): the selection grids, the model and calibrator alternatives, the arena
   strategies and the ablation variants. The log is upserted by trial id, never truncated.

Lives in `src/sim` because it replays execution prices (R2: only `src/sim` reads open(D));
`python -m src.strategies.arena` is the entry point the Makefile names. Wording follows R11:
returns *coincided with* signals; nothing here is a prediction or advice.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import splits
from src.collectors.common import ROOT
from src.marketview import MarketView
from src.schemas import DISCLAIMER, Signal
from src.sim import metrics as m
from src.sim.backtest import DEV_SPLITS, Run, resolve_split, run_simulation
from src.sim.controls import random_controls
from src.sim.impact_train import val_b_range
from src.sim.run import _git_sha
from src.strategies.base import backtest_config, make_signal, strategy_params
from src.strategies.registry import BENCHMARK, build
from src.strategies.signal_stack import AggregateEqual, model_dir

RESULTS = ROOT / "evals" / "results"
DOC = ROOT / "docs" / "05_arena.md"
DOC_FINAL = ROOT / "docs" / "06_final_run.md"  # written once, by the final run (D-43)
WHY = (
    "the deflated Sharpe asks whether the headline strategy's Sharpe beats what the best of all "
    "N configurations we tried would reach by luck; a plain Sharpe ignores that search"
)
TRIAL_FIELDS = [
    "trial_id", "family", "strategy", "params", "period", "start", "end", "metric", "value",
    "sharpe_ann", "n_days", "round_trips", "source", "first_logged_at", "last_logged_at",
]  # fmt: skip


def arena_config() -> dict[str, Any]:
    return strategy_params("arena")


class AblationReplay:
    """`ablation_of` with a subset of its components, from recorded component scores."""

    def __init__(
        self,
        parts: dict[tuple[str, date], dict[str, float]],
        components: list[str],
        hold: int,
        threshold: float,
        name: str,
    ) -> None:
        self.name = name
        self.parts, self.components, self.threshold = parts, components, threshold
        self.horizon_days = self.min_hold_days = hold

    def signal(self, ticker: str, view: MarketView) -> Signal:
        p = self.parts.get((ticker, view.as_of.date()))
        s = float(np.mean([p[c] for c in self.components])) if p else 0.0
        # same rule as the stacked strategies: below the entry threshold the suggestion is cash
        return make_signal(self.name, ticker, view, s if s > self.threshold else min(s, 0.0))


def _r(x: Any, nd: int = 4) -> Any:
    if isinstance(x, float):
        return round(x, nd) if math.isfinite(x) else None
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    return x


def _returns(run: Run) -> pd.Series:
    r = m.daily_returns(run.sim.equity, run.cfg.initial_capital)
    return pd.Series(r, index=pd.Index(run.sim.days, name="day"))


def _boot_kw() -> dict[str, Any]:
    b = backtest_config()["bootstrap"]
    return {"n_resamples": int(b["n_resamples"]), "block_days": int(b["block_days"]),
            "ci": float(b["ci"]), "seed": int(b["seed"])}  # fmt: skip


def choice_period(name: str) -> str:
    """Where the parameters of `name` were chosen (R3; D-35, D-40)."""
    sel = strategy_params("selection")["period"]
    if name in sel:
        extra = " (+ impact model on train, calibrator on val_a)" if name != "event" else ""
        return f"{sel[name]}{extra}"
    if name == "finbert_v0":
        return "fixed a priori (D-36)"
    return "fixed a priori (config, D-35)"


def leaderboard_row(name: str, run: Run, val_b_start: date | None) -> dict[str, Any]:
    res, r, dpy = run.result, _returns(run), int(backtest_config()["trading_days_per_year"])
    rb = r[r.index >= val_b_start].to_numpy() if val_b_start else r.iloc[:0].to_numpy()
    choice = choice_period(name)
    return {
        "strategy": name,
        "sharpe": res.sharpe,
        "sharpe_ci95": res.sharpe_ci95,
        "total_return": res.total_return,
        "max_drawdown": res.max_drawdown,
        "n_trades": res.n_trades,
        "round_trips": len(run.sim.round_trip_pnls),
        "hit_rate": res.hit_rate,
        "turnover": res.turnover,
        "costs_paid": res.costs_paid,
        "mean_exposure": float(np.mean(run.sim.exposure)) if run.sim.exposure else 0.0,
        "risk_managed": run.risk_managed,
        "choice_period": choice,
        "validation_in_sample": "val_a" in choice,
        "sharpe_val_b": m.sharpe(rb, dpy),
        "sharpe_val_b_ci95": m.bootstrap_ci(rb, m.sharpe_stat(dpy), **_boot_kw()),
        "n_days": res.n_days,
        "n_days_val_b": int(rb.size),
    }


def ablation(full: Run, parts: dict, split: str, db_path: Path | None) -> list[dict[str, Any]]:
    cfg = arena_config()
    name = cfg["ablation_of"]
    comps = list(strategy_params(name)["components"])
    chosen = json.loads((model_dir() / "selected.json").read_text(encoding="utf-8"))[name]
    hold, thr = int(chosen["chosen"]["hold_days"]), float(chosen["chosen"]["entry_threshold"])
    rows = [{"variant": "all components", "dropped": None, "sharpe": full.result.sharpe,
             "sharpe_ci95": full.result.sharpe_ci95, "total_return": full.result.total_return,
             "round_trips": len(full.sim.round_trip_pnls)}]  # fmt: skip
    for drop in comps:
        keep = [c for c in comps if c != drop]
        strat = AblationReplay(parts, keep, hold, thr, f"without_{drop}")
        run = run_simulation(strat, split, db_path, risk_managed=full.risk_managed,
                             prices=full.prices)  # fmt: skip
        rows.append({"variant": strat.name, "dropped": drop, "sharpe": run.result.sharpe,
                     "sharpe_ci95": run.result.sharpe_ci95,
                     "total_return": run.result.total_return,
                     "round_trips": len(run.sim.round_trip_pnls)})  # fmt: skip
    base = full.result.sharpe or 0.0
    for row in rows:
        row["delta_sharpe_vs_all"] = (row["sharpe"] or 0.0) - base
    return rows


# --- trials log ----------------------------------------------------------------------------


def _trial(**kw: Any) -> dict[str, Any]:
    row = dict.fromkeys(TRIAL_FIELDS, "")
    row.update({k: ("" if v is None else v) for k, v in kw.items()})
    return row


def trial_rows(
    board: list[dict[str, Any]],
    ablations: list[dict[str, Any]],
    split_range: tuple[date, date],
    selected: dict[str, Any] | None,
    impact_report: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Every configuration this repo has looked at, with its Sharpe where it has one."""
    lo, hi = (d.isoformat() for d in split_range)
    rows = []
    for b in board:
        rows.append(_trial(
            trial_id=f"arena:{b['strategy']}:validation", family="arena", strategy=b["strategy"],
            params="config/strategies.yaml + selected.json", period="validation", start=lo,
            end=hi, metric="sharpe", value=b["sharpe"], sharpe_ann=b["sharpe"],
            n_days=b["n_days"], round_trips=b["round_trips"], source="make arena",
        ))  # fmt: skip
    for a in ablations[1:]:
        rows.append(_trial(
            trial_id=f"ablation:{a['variant']}:validation", family="ablation",
            strategy=a["variant"], params=json.dumps({"dropped": a["dropped"]}),
            period="validation", start=lo, end=hi, metric="sharpe", value=a["sharpe"],
            sharpe_ann=a["sharpe"], round_trips=a["round_trips"], source="make arena",
        ))  # fmt: skip
    for name, s in (selected or {}).items():
        for g in s["grid"]:
            params = {"hold_days": g["hold_days"], "entry_threshold": g["entry_threshold"]}
            rows.append(_trial(
                trial_id=f"select:{name}:{s['period']}:h{g['hold_days']}:t{g['entry_threshold']}",
                family="selection_grid", strategy=name, params=json.dumps(params),
                period=s["period"], start=s["start"], end=s["end"], metric="sharpe",
                value=g["sharpe"], sharpe_ann=g["sharpe"], round_trips=g["round_trips"],
                source="make train (src/sim/select.py)",
            ))  # fmt: skip
    if impact_report:
        for model in ("logreg", "hgb"):
            if model in impact_report:
                rows.append(_trial(
                    trial_id=f"impact_model:{model}", family="impact_model", strategy="impact",
                    params=json.dumps({"model": model}), period="val_a", metric="auc",
                    value=impact_report[model]["val_a"]["auc"],
                    source="make train (src/sim/impact_train.py)",
                ))  # fmt: skip
        for cal in ("platt", "isotonic"):
            if cal in impact_report.get("calibration", {}):
                rows.append(_trial(
                    trial_id=f"calibrator:{cal}", family="calibrator", strategy="impact",
                    params=json.dumps({"calibrator": cal}), period="val_a", metric="see report",
                    source="make train (src/sim/impact_train.py)",
                ))  # fmt: skip
    return rows


def read_trials(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def upsert_trials(path: Path, rows: list[dict[str, Any]], now: str) -> list[dict[str, Any]]:
    """Merge `rows` into the CSV by trial id: keep first_logged_at, refresh everything else.
    Trials logged by other runs (absent from `rows`) are kept: the log only grows."""
    old = {r["trial_id"]: r for r in read_trials(path)}
    for r in rows:
        prev = old.get(r["trial_id"])
        r = {k: _r(v) if isinstance(v, float) else v for k, v in r.items()}
        r["first_logged_at"] = prev["first_logged_at"] if prev else now
        r["last_logged_at"] = now
        old[r["trial_id"]] = r
    out = [old[k] for k in sorted(old)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TRIAL_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    return out


def trial_sharpes_daily(trials: list[dict[str, Any]], days_per_year: int) -> list[float]:
    out = []
    for t in trials:
        try:
            v = float(t["sharpe_ann"])
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out.append(v / math.sqrt(days_per_year))
    return out


# --- the run -------------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run_arena(
    split: str = "validation",
    db_path: Path | None = None,
    n_random: int | None = None,
    runs_out: dict[str, Run] | None = None,
) -> dict[str, Any]:
    """The arena on `split`. On the test split (reachable only in the final run, D-43) there
    is no val_b column and the trials log is read, not extended: the final run is not a trial.
    `runs_out` receives the simulations (the final run writes backtest records from them)."""
    split = resolve_split(split)
    is_final = split == splits.FINAL_SPLIT
    cfg = arena_config()
    dpy = int(backtest_config()["trading_days_per_year"])
    val_b_start = None if is_final else val_b_range(db_path)[0]
    runs: dict[str, Run] = {}
    parts: dict[tuple[str, date], dict[str, float]] = {}
    prices = None
    for name in cfg["strategies"]:
        strat = build(name)
        if isinstance(strat, AggregateEqual) and name == cfg["ablation_of"]:
            strat.parts_log = parts
        runs[name] = run_simulation(strat, split, db_path, prices=prices)
        prices = runs[name].prices
        print(f"{name}: sharpe {runs[name].result.sharpe}")
    board = [leaderboard_row(n, r, val_b_start) for n, r in runs.items()]
    corr = pd.DataFrame({n: _returns(r) for n, r in runs.items()}).corr()

    abl = ablation(runs[cfg["ablation_of"]], parts, split, db_path)

    rc = cfg["random_controls"]
    controls: dict[str, dict[str, Any]] = {}
    for name, run in runs.items():
        if name == BENCHMARK:
            continue
        controls[name] = {
            mode: random_controls(
                run.sim, run.tickers, run.prices, run.cfg, n=int(n_random or rc["n"]),
                seed=int(rc["seed"]), days_per_year=dpy, mode=mode,
            )
            for mode in rc["modes"]
        }  # fmt: skip
        pct = {mode: c["percentile"] for mode, c in controls[name].items()}
        print(f"{name}: random-control percentiles {pct}")

    days = runs[cfg["strategies"][0]].sim.days
    now = datetime.now(UTC).isoformat(timespec="seconds")
    if is_final:
        trials = read_trials(ROOT / cfg["trials_log"])
    else:
        trials = upsert_trials(
            ROOT / cfg["trials_log"],
            trial_rows(board, abl, (days[0], days[-1]), _load_json(model_dir() / "selected.json"),
                       _load_json(model_dir() / "impact_report.json")),
            now,
        )  # fmt: skip
    if runs_out is not None:
        runs_out.update(runs)
    trial_sr = trial_sharpes_daily(trials, dpy)
    dsr = {n: m.deflated_sharpe(_returns(r).to_numpy(), trial_sr, len(trials))
           for n, r in runs.items()}  # fmt: skip
    return {
        "split": split,
        "start": days[0].isoformat(),
        "end": days[-1].isoformat(),
        "val_b_start": val_b_start.isoformat() if val_b_start else None,
        "n_days": len(days),
        "leaderboard": board,
        "correlation": {a: {b: corr.loc[a, b] for b in corr.columns} for a in corr.index},
        "ablation": abl,
        "random_controls": controls,
        "deflated_sharpe": dsr,
        "trials": {
            "n": len(trials),
            "n_with_sharpe": len(trial_sr),
            "by_family": pd.Series([t["family"] for t in trials])
            .value_counts()
            .astype(int)
            .to_dict(),
            "path": cfg["trials_log"],
        },  # fmt: skip
        "run_at": now,
    }


def component(split: str) -> str:
    """`arena` for validation, `arena_<split>` otherwise (the final run: never overwrites)."""
    return "arena" if split == "validation" else f"arena_{split}"


def doc_path(split: str) -> Path:
    return DOC if split == "validation" else DOC_FINAL


def record(arena: dict[str, Any]) -> dict[str, Any]:
    cfg = arena_config()
    head = cfg["headline"]
    target = float(cfg["dsr_target"])
    d = arena["deflated_sharpe"][head]
    value = d["dsr"] if d["dsr"] is not None else 0.0
    rc = (
        arena["random_controls"]
        .get(head, {})
        .get(arena_config()["random_controls"]["modes"][0], {})
    )
    row = next(b for b in arena["leaderboard"] if b["strategy"] == head)
    ann = math.sqrt(int(backtest_config()["trading_days_per_year"]))
    pct = rc.get("percentile")
    on = arena["split"]
    if arena.get("val_b_start"):
        on += f" (val_b only {_f(row['sharpe_val_b'])})"
    notes = (
        f"{head}: Sharpe {_f(row['sharpe'])} on {on}; deflated Sharpe {value:.3f} with N = "
        f"{arena['trials']['n']} trials (SR0 {d['sr0_daily'] * ann:.2f} annualised); "
        f"random-control percentile {_f(pct, 1)} (same entry days, random names)."
    )
    if value < target:
        notes += (
            " Below target: the Sharpe is not distinguishable from the best of N unskilled "
            f"configurations. See docs/{doc_path(arena['split']).name}."
        )
    return {
        "component": component(arena["split"]),
        "metric": f"deflated_sharpe_{head}",
        "value": round(value, 4),
        "target": target,
        "why_this_metric": WHY,
        "gold_set": (
            f"derived: {arena['trials']['n']} trials in {arena['trials']['path']}; "
            "matched random controls"
        ),
        "n": arena["n_days"],
        "split": arena["split"],
        "model": "rules + impact model (data/models)",
        "git_sha": _git_sha(),
        "run_at": arena["run_at"],
        "notes": notes,
        "details": {**_r(arena), "disclaimer": DISCLAIMER},
    }


# --- docs/05_arena.md ----------------------------------------------------------------------


def _f(x: Any, nd: int = 2) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def _ci(ci: Any) -> str:
    return "–" if not ci else f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def render(rec: dict[str, Any]) -> str:
    a = rec["details"]
    head = arena_config()["headline"]
    ann = math.sqrt(int(backtest_config()["trading_days_per_year"]))
    lines = [
        f"# Strategy arena ({a['split']})",
        "",
        f"Generated by `src/sim/arena.py` (D-42) from `evals/results/{rec['component']}.json`; "
        "do not edit by hand.",
        "",
        f"> {DISCLAIMER}",
        "",
        f"Split **{a['split']}** {a['start']} .. {a['end']} ({a['n_days']} sessions, watchlist, "
        f"5 bps per side). val_b = sessions from {a['val_b_start']}. git `{rec['git_sha']}`, run "
        f"{rec['run_at']}. Returns *coincided with* the signals; none of this is a prediction.",
        "",
        f"**Headline:** {rec['notes']}",
        "",
        "## 1. Leaderboard",
        "",
        "Sharpe after costs, 95 % moving-block bootstrap CI (block 5 days). *Chosen on* says where "
        "the strategy's parameters were picked: rows chosen on val_a are partly in-sample on "
        "validation, so read their val_b column.",
        "",
        "| strategy | Sharpe | 95 % CI | val_b Sharpe | val_b CI | return | max DD | round "
        "trips | exposure | costs $ | chosen on |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for b in a["leaderboard"]:
        lines.append(
            f"| {b['strategy']} | {_f(b['sharpe'])} | {_ci(b['sharpe_ci95'])} | "
            f"{_f(b['sharpe_val_b'])} | {_ci(b['sharpe_val_b_ci95'])} | "
            f"{_f(100 * b['total_return'], 1)} % | {_f(100 * b['max_drawdown'], 1)} % | "
            f"{b['round_trips']} | {_f(100 * b['mean_exposure'], 0)} % | "
            f"{_f(b['costs_paid'], 0)} | {b['choice_period']} |"
        )
    names = list(a["correlation"])
    lines += [
        "",
        "## 2. Correlation of daily returns",
        "",
        "Pearson correlation of daily strategy returns (cash days count as 0 % return).",
        "",
        "| | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for r in names:
        lines.append(f"| {r} | " + " | ".join(_f(a["correlation"][r][c]) for c in names) + " |")
    lines += [
        "",
        f"## 3. Drop-one ablation of `{arena_config()['ablation_of']}`",
        "",
        "Component scores recorded during the full run, replayed without one component (same "
        "hold period, entry threshold and risk manager). A negative delta means the dropped "
        "component coincided with better returns on validation.",
        "",
        "| variant | dropped | Sharpe | 95 % CI | return | round trips | Δ Sharpe vs all |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for r in a["ablation"]:
        lines.append(
            f"| {r['variant']} | {r['dropped'] or '–'} | {_f(r['sharpe'])} | "
            f"{_ci(r['sharpe_ci95'])} | {_f(100 * r['total_return'], 1)} % | "
            f"{r['round_trips']} | {_f(r['delta_sharpe_vs_all'])} |"
        )
    lines += [
        "",
        "## 4. Random controls (matched exposure and trade count)",
        "",
        "Each control keeps the strategy's round trips (count, length in sessions, open or close "
        "exit) and runs through the same engine, sizing and costs, without the risk manager "
        "(`src/sim/controls.py`). *names*: same entry sessions, random tickers - same number of "
        "positions on every day, so exposure and trade count match; it asks whether the choice of "
        "*what* to buy beat chance. *timing+names*: entry sessions random too; a dense strategy "
        "may not fit completely (achieved counts shown). Percentile = share of controls with a "
        "lower Sharpe.",
        "",
        "| strategy | control | n | Sharpe | percentile | random p05 | median | p95 | round trips "
        "(ours / random) | exposure (ours / random) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for n, by_mode in a["random_controls"].items():
        for mode, c in by_mode.items():
            label = mode.replace("timing_and_names", "timing+names")
            if not c.get("round_trips"):
                lines.append(f"| {n} | {label} | 0 | {_f(c['sharpe'])} | – | – | – | – | 0 | – |")
                continue
            lines.append(
                f"| {n} | {label} | {c['n']} | {_f(c['sharpe'])} | {_f(c['percentile'], 1)} | "
                f"{_f(c['random_sharpe_p05'])} | {_f(c['random_sharpe_median'])} | "
                f"{_f(c['random_sharpe_p95'])} | {c['round_trips']} / "
                f"{_f(c['random_round_trips_mean'], 1)} | {_f(100 * c['mean_exposure'], 0)} % / "
                f"{_f(100 * c['random_mean_exposure'], 0)} % |"
            )
    t = a["trials"]
    fam = ", ".join(f"{k} {v}" for k, v in t["by_family"].items())
    lines += [
        "",
        "## 5. Deflated Sharpe",
        "",
        f"N = **{t['n']}** configurations in `{t['path']}` ({fam}); the variance of the trial "
        f"Sharpes uses the {t['n_with_sharpe']} trials that have one (selection grids on train / "
        "val_a, arena and ablation rows on validation). SR0 = expected maximum Sharpe of N "
        "unskilled trials; DSR = P(true Sharpe > SR0) given skew and kurtosis of the daily "
        f"returns. Target {rec['target']}.",
        "",
        "| strategy | Sharpe (ann.) | SR0 (ann.) | skew | kurtosis | PSR vs 0 | DSR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for n, d in a["deflated_sharpe"].items():
        mark = " **(headline)**" if n == head else ""
        lines.append(
            f"| {n}{mark} | {_f((d['sharpe_daily'] or 0) * ann)} | "
            f"{_f((d['sr0_daily'] or 0) * ann)} | {_f(d['skew'])} | "
            f"{_f(d['kurtosis'])} | {_f(d['psr_vs_zero'], 3)} | {_f(d['dsr'], 3)} |"
        )
    lines += [
        "",
        "## Caveats",
        "",
        f"- {a['n_days']} sessions is short: every CI above is wide, and one regime "
        "(summer 2026) dominates. Buy-and-hold on a 10-name watchlist chosen with "
        "hindsight is a strong, survivorship-flattered benchmark (L-07).",
        (
            "- `impact` and `aggregate_equal` chose their calibrator and hold/threshold on "
            "val_a; their full-validation Sharpe is partly in-sample. The val_b column is the "
            "honest one."
            if a.get("val_b_start")
            else "- This is the single test-period run (R3): every choice was made before it, "
            "nothing may be tuned after it."
        ),
        "- Random controls run without the risk manager (the stacked strategies run with it), so "
        "a stop that cut a losing trip short is kept as a shorter trip, not as a rule.",
        "- The masking experiment (LLM memory check) was not run in this session.",
        "",
    ]
    return "\n".join(lines)


def write(rec: dict[str, Any]) -> tuple[Path, Path]:
    path = RESULTS / f"{rec['component']}.json"
    path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8", newline="\n")
    doc = doc_path(rec["split"])
    doc.write_text(render(rec), encoding="utf-8", newline="\n")
    return path, doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="validation", choices=sorted(DEV_SPLITS))
    ap.add_argument("--n-random", type=int, default=None, help="override random_controls.n")
    ap.add_argument("--final", action="store_true", help="the single test-period run (D-43)")
    args = ap.parse_args(argv)
    if args.final:
        from src.sim.final import run_final  # late import: final imports this module

        run_final()
        return 0
    rec = record(run_arena(args.split, n_random=args.n_random))
    p, d = write(rec)
    print(f"arena: {rec['metric']} = {rec['value']} (target {rec['target']}) -> "
          f"{p.relative_to(ROOT)}, {d.relative_to(ROOT)}")  # fmt: skip
    from evals.run_all import refresh_doc  # late import: evals is a consumer of src

    refresh_doc()
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
