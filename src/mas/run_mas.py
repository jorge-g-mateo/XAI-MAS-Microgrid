"""Experiment: impact of negotiation strategies + comparison with the optimum.

Runs every (solar_strategy x wind_strategy) combination on one scenario, records
the negotiated outcome, and (optionally) compares each outcome against the
multi-objective Pareto front from the optimization module — answering "how close
does emergent negotiation get to the optimal market allocation, and how do
information hiding / deception / opponent modeling shift it?".

Usage:
    python -m src.mas.run_mas --timestamp "2017-06-15 19:00:00"
    python -m src.mas.run_mas --row 4000 --no-optimum
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.mas.simulation import run_negotiation
from src.mas.strategies import STRATEGIES
from src.optimization.market import scenario_from_timestamp

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def _dominated_by_front(point, front: np.ndarray) -> tuple[bool, float]:
    """Is the negotiated (profit_solar, profit_wind, buyer_cost) point dominated by
    any Pareto point? Also return the min normalized distance to the front.

    Maximize profits, minimize cost. Front columns: [profit_solar, profit_wind, cost].

    Domination requires a margin of ``tol`` (relative to each objective's
    magnitude): with zero marginal cost the market satisfies
    cost = profit_solar + profit_wind on every demand-covering point, so true
    domination between feasible points is impossible and any strict flag at
    float precision is rounding noise, not a real efficiency loss.
    """
    ps, pw, cost = point
    tol = 1e-6 * (np.abs(front).mean(axis=0) + 1.0)
    dominates = (
        (front[:, 0] >= ps - tol[0]) & (front[:, 1] >= pw - tol[1])
        & (front[:, 2] <= cost + tol[2])
        & ((front[:, 0] > ps + tol[0]) | (front[:, 1] > pw + tol[1])
           | (front[:, 2] < cost - tol[2]))
    )
    scale = front.std(axis=0) + 1e-9
    dist = np.linalg.norm((front - np.array(point)) / scale, axis=1).min()
    return bool(dominates.any()), float(dist)


def _grid(cfg, rounds: int, buyer_strategy: str) -> pd.DataFrame:
    """The full 6x6 seller x seller grid for one fixed consumer strategy."""
    names = list(STRATEGIES)
    rows = []
    for s in names:
        for w in names:
            res = run_negotiation(cfg, s, w, buyer_strategy=buyer_strategy,
                                  rounds=rounds, log=False)
            rows.append({
                "buyer_strategy": buyer_strategy,
                "solar_strategy": s, "wind_strategy": w,
                "profit_solar": res.profit_solar, "profit_wind": res.profit_wind,
                "buyer_cost": res.buyer_cost, "shortfall": res.shortfall,
                "price_solar": res.final_price_solar, "price_wind": res.final_price_wind,
            })
    return pd.DataFrame(rows)


def _compare_to_front(df: pd.DataFrame, front: np.ndarray) -> pd.DataFrame:
    dom, dist = [], []
    for _, r in df.iterrows():
        d, dd = _dominated_by_front(
            (r.profit_solar, r.profit_wind, r.buyer_cost), front)
        dom.append(d); dist.append(dd)
    df = df.copy()
    df["dominated_by_optimum"] = dom
    df["dist_to_pareto"] = np.round(dist, 3)
    return df


def run_experiment(cfg, rounds: int = 12, with_optimum: bool = True,
                   buyer_strategy: str = "price_taker") -> pd.DataFrame:
    """One 6x6 seller grid for a single consumer strategy (legacy entry point)."""
    df = _grid(cfg, rounds, buyer_strategy)
    if with_optimum:
        from src.optimization.problem import decode_objectives
        from src.optimization.run_optimization import run as run_opt
        _, _, combined = run_opt(cfg, runs=2, evals=6000)
        front = np.array([decode_objectives(s.objectives) for s in combined])
        df = _compare_to_front(df, front)
    return df


def run_consumer_study(cfg, rounds: int = 12, with_optimum: bool = True,
                       buyers: list[str] | None = None) -> pd.DataFrame:
    """The 6x6 seller game re-played under EACH consumer strategy, all projected
    against the SAME Pareto front (computed once). This is the unified comparison:
    the consumer is a first-class third agent whose strategy shifts the whole game,
    measured on the same axes as the optimum (Workstream A/D)."""
    from src.mas.buyer_strategies import BUYER_STRATEGIES
    buyers = buyers if buyers is not None else list(BUYER_STRATEGIES)

    front = None
    if with_optimum:
        from src.optimization.problem import decode_objectives
        from src.optimization.run_optimization import run as run_opt
        _, _, combined = run_opt(cfg, runs=2, evals=6000)  # buyer-independent → once
        front = np.array([decode_objectives(s.objectives) for s in combined])

    parts = []
    for b in buyers:
        g = _grid(cfg, rounds, b)
        if front is not None:
            g = _compare_to_front(g, front)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--timestamp", type=str)
    g.add_argument("--row", type=int)
    ap.add_argument("--demand", type=float, default=None)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--no-optimum", action="store_true",
                    help="skip the Pareto-front comparison (faster)")
    ap.add_argument("--single-buyer", type=str, default=None,
                    help="run only one consumer strategy (legacy single 6x6 grid)")
    args = ap.parse_args()

    ts = args.timestamp if args.timestamp is not None else (args.row if args.row is not None else "2017-06-15 19:00:00")
    cfg = scenario_from_timestamp(ts, demand=args.demand)
    print(f"Scenario {cfg.label}: demand={cfg.demand:.2f} "
          f"solar={cfg.gen_solar:.2f} wind={cfg.gen_wind:.2f} kW\n")

    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)

    if args.single_buyer:
        df = run_experiment(cfg, rounds=args.rounds, with_optimum=not args.no_optimum,
                            buyer_strategy=args.single_buyer)
        out = RESULTS / "mas_strategy_comparison.csv"
        df.to_csv(out, index=False)
        print(df.round(2).to_string(index=False))
        print(f"\nSaved -> {out}")
        return

    # Default: the unified consumer study (6x6 sellers under every consumer strategy).
    df = run_consumer_study(cfg, rounds=args.rounds, with_optimum=not args.no_optimum)
    out = RESULTS / "mas_consumer_grid.csv"
    df.to_csv(out, index=False)
    print(f"Saved 6x6 x {df.buyer_strategy.nunique()} buyers -> {out}\n")

    # Headline: how the consumer strategy shifts the honest/honest social optimum.
    print("How the CONSUMER strategy shifts outcomes (honest/honest cell):")
    for b, g in df.groupby("buyer_strategy"):
        hh = g[(g.solar_strategy == "honest") & (g.wind_strategy == "honest")].iloc[0]
        print(f"  buyer={b:24s} cost={hh.buyer_cost:7.1f} short={hh.shortfall:.2f} "
              f"pS={hh.price_solar:.2f} pW={hh.price_wind:.2f}")
    print("\nMutual opponent_modeling cell (surplus extraction):")
    for b, g in df.groupby("buyer_strategy"):
        oo = g[(g.solar_strategy == "opponent_modeling")
               & (g.wind_strategy == "opponent_modeling")].iloc[0]
        print(f"  buyer={b:24s} cost={oo.buyer_cost:7.1f} short={oo.shortfall:.2f} "
              f"pS={oo.price_solar:.2f} pW={oo.price_wind:.2f}")


if __name__ == "__main__":
    main()
