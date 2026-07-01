"""Many-objective optimization with a storage arbitrageur (M1, optimization side).

Solves the 4-objective ``EnergyMarketBatteryProblem`` — maximize solar profit,
wind profit and battery profit, minimize consumer cost — with **NSGA-III** (the
algorithm motivated precisely for many-objective problems via reference
directions). The battery's profit comes from a resale markup ``f ∈ [1, 1.2]`` on
the unsold surplus, a **proxy for the temporal spread** (the real arbitrage is
validated in ``src/mas/battery.py``).

This is a SEPARATE variant: the 3-objective optimization (the clean Pareto
benchmark, the 15-seed stats and the structural finding) is left untouched.

Outputs (results/):
  - opt_battery_front.csv       the 4-objective non-dominated front + variables
  - opt_battery_parallel.png    parallel-coordinates view of the 4 objectives
  - opt_battery_tradeoff.png    battery profit vs the other objectives

Usage:
    python -m src.optimization.run_optimization_battery --timestamp "2017-06-15 19:00:00"
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

logging.getLogger("jmetal").setLevel(logging.WARNING)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII, UniformReferenceDirectionFactory
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

from src.optimization.market import scenario_from_timestamp
from src.optimization.problem import EnergyMarketBatteryProblem, decode_objectives_battery

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
LABELS = ["Profit Solar", "Profit Wind", "Buyer Cost", "Battery Profit"]


def run(timestamp: str, runs: int = 3, evals: int = 12000, n_points: int = 120,
        seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    cfg = scenario_from_timestamp(timestamp, enable_battery=True)
    print(f"Scenario {cfg.label}: demand={cfg.demand:.2f} solar={cfg.gen_solar:.2f} "
          f"wind={cfg.gen_wind:.2f} kW")

    solutions = []
    for _ in range(runs):
        problem = EnergyMarketBatteryProblem(cfg)
        ref = UniformReferenceDirectionFactory(4, n_points=n_points)
        algo = NSGAIII(
            reference_directions=ref, problem=problem, population_size=n_points,
            mutation=PolynomialMutation(probability=1.0 / 6, distribution_index=20),
            crossover=SBXCrossover(probability=1.0, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max_evaluations=evals))
        algo.run()
        feasible = [s for s in algo.result() if s.constraints[0] >= -1e-9]
        solutions.extend(get_non_dominated_solutions(feasible))

    front = get_non_dominated_solutions(solutions)
    rows = []
    for s in front:
        ps, pw, cost, batt = decode_objectives_battery(s.objectives)
        q_solar, q_wind, p_solar, p_wind, q_batt, markup = s.variables
        rows.append({"profit_solar": ps, "profit_wind": pw, "buyer_cost": cost,
                     "battery_profit": batt, "q_solar": q_solar, "q_wind": q_wind,
                     "p_solar": p_solar, "p_wind": p_wind, "q_batt": q_batt,
                     "markup": markup})
    return pd.DataFrame(rows).round(3)


def save_outputs(df: pd.DataFrame, timestamp: str):
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "opt_battery_front.csv", index=False)
    obj = df[["profit_solar", "profit_wind", "buyer_cost", "battery_profit"]].to_numpy()

    # Parallel coordinates (normalized per objective).
    lo, hi = obj.min(axis=0), obj.max(axis=0)
    span = np.where(hi - lo < 1e-9, 1.0, hi - lo)
    norm = (obj - lo) / span
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = range(4)
    for row in norm:
        ax.plot(xs, row, color="tab:blue", alpha=0.15)
    ax.set_xticks(list(xs)); ax.set_xticklabels(LABELS)
    for i in xs:
        ax.axvline(i, color="grey", lw=0.6)
        ax.text(i, 1.02, f"[{lo[i]:.0f}, {hi[i]:.0f}]", ha="center", fontsize=7, color="grey")
    ax.set_ylabel("normalized objective value")
    ax.set_title(f"4-objective front with a battery arbitrageur (NSGA-III) — {timestamp}")
    fig.tight_layout(); fig.savefig(RESULTS / "opt_battery_parallel.png", dpi=130)
    plt.close(fig)

    # Battery profit vs the other three objectives (is it a real trade-off?).
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, lab in zip(axes, ["profit_solar", "profit_wind", "buyer_cost"],
                            LABELS[:3]):
        ax.scatter(df[col], df["battery_profit"], s=12, alpha=0.5, color="tab:purple")
        c = np.corrcoef(df[col], df["battery_profit"])[0, 1] if len(df) > 1 else float("nan")
        ax.set_xlabel(lab); ax.set_ylabel("Battery Profit")
        ax.set_title(f"r = {c:+.2f}"); ax.grid(True, alpha=0.3)
    fig.suptitle("Battery profit vs. the other objectives (correlation = trade-off strength)")
    fig.tight_layout(); fig.savefig(RESULTS / "opt_battery_tradeoff.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamp", default="2017-06-15 19:00:00")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--evals", type=int, default=12000)
    args = ap.parse_args()

    df = run(args.timestamp, runs=args.runs, evals=args.evals)
    save_outputs(df, args.timestamp)
    print(f"\nFront size: {len(df)} non-dominated solutions")
    print(df.describe().loc[["min", "max"]].round(2).to_string())
    if len(df) > 1:
        print("\nCorrelation of battery_profit with:")
        for col in ["profit_solar", "profit_wind", "buyer_cost"]:
            print(f"  {col:14s}: r = {np.corrcoef(df[col], df['battery_profit'])[0,1]:+.3f}")
    print(f"\nOutputs -> {RESULTS}/ (opt_battery_*.{{csv,png}})")


if __name__ == "__main__":
    main()
