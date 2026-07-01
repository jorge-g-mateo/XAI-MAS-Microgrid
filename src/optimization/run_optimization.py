"""Run and compare NSGA-II, NSGA-III and SPEA2 on the energy-market problem.

Outputs (written to ``results/``):
  - ``metrics_summary.csv``           quality indicators per algorithm (HV/GD/IGD/Eps)
  - ``superfront_<ALGO>.csv``         non-dominated superfront per algorithm
  - ``pareto_front.png``              2D projection (profit AS vs AE, colored by cost)

Usage:
    python -m src.optimization.run_optimization --timestamp "2017-06-15 13:00:00"
    python -m src.optimization.run_optimization --row 4000 --runs 5 --evals 10000
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

# jMetal logs every generation at DEBUG; keep the pipeline output readable.
logging.getLogger("jmetal").setLevel(logging.WARNING)

import matplotlib

matplotlib.use("Agg")  # headless backend for Docker/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jmetal.algorithm.multiobjective.moead import MOEAD
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.nsgaiii import (
    NSGAIII,
    UniformReferenceDirectionFactory,
)
from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.core.quality_indicator import (
    EpsilonIndicator,
    GenerationalDistance,
    HyperVolume,
    InvertedGenerationalDistance,
)
from jmetal.operator.crossover import DifferentialEvolutionCrossover, SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.util.aggregation_function import Tschebycheff
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

from src.optimization.market import MarketConfig, scenario_from_timestamp
from src.optimization.metrics import spread
from src.optimization.problem import EnergyMarketProblem, decode_objectives

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ALGORITHMS = ["NSGAII", "NSGAIII", "SPEA2", "MOEAD"]

# MOEA/D decomposes the 3-objective problem into one scalar subproblem per
# weight vector, so its population size equals the number of weight vectors.
# We use the Das-Dennis simplex lattice with H=12 -> C(14,2)=91 vectors, the
# same count as NSGA-III's 91 uniform reference directions (fair comparison).
MOEAD_POP = 91
_WEIGHTS_DIR = Path(__file__).resolve().parent / "moead_weights"


def _ensure_moead_weights(m: int = 3, h: int = 12) -> Path:
    """Write the Das-Dennis simplex-lattice weight file jMetalPy expects
    (``W<m>D_<n>.dat``) if it does not exist yet; return its directory."""
    from itertools import combinations
    from math import comb

    n = comb(h + m - 1, m - 1)
    path = _WEIGHTS_DIR / f"W{m}D_{n}.dat"
    if not path.is_file():
        _WEIGHTS_DIR.mkdir(exist_ok=True)
        lines = []
        # Das-Dennis: all compositions of h into m non-negative integers / h.
        for cuts in combinations(range(h + m - 1), m - 1):
            prev, parts = -1, []
            for c in (*cuts, h + m - 1):
                parts.append(c - prev - 1)
                prev = c
            lines.append(" ".join(f"{p / h:.6f}" for p in parts))
        path.write_text("\n".join(lines) + "\n")
    return _WEIGHTS_DIR


def _build(name: str, problem: EnergyMarketProblem, pop: int, evals: int):
    nvars = problem.number_of_variables()
    mutation = PolynomialMutation(probability=1.0 / nvars, distribution_index=20)
    crossover = SBXCrossover(probability=1.0, distribution_index=20)
    term = StoppingByEvaluations(max_evaluations=evals)
    if name == "NSGAII":
        return NSGAII(problem=problem, population_size=pop, offspring_population_size=pop,
                      mutation=mutation, crossover=crossover, termination_criterion=term)
    if name == "NSGAIII":
        return NSGAIII(reference_directions=UniformReferenceDirectionFactory(3, n_points=91),
                       problem=problem, population_size=pop,
                       mutation=mutation, crossover=crossover, termination_criterion=term)
    if name == "SPEA2":
        return SPEA2(problem=problem, population_size=pop, offspring_population_size=pop,
                     mutation=mutation, crossover=crossover, termination_criterion=term)
    if name == "MOEAD":
        # Decomposition paradigm (vs. the dominance ranking of the other three):
        # Tschebycheff aggregation with an adaptive ideal point, neighbourhood
        # mating (T=20), and the standard MOEA/D-DE setting CR=1.0, F=0.5,
        # delta=0.9, nr=2.
        return MOEAD(problem=problem, population_size=MOEAD_POP,
                     mutation=mutation,
                     crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5),
                     aggregation_function=Tschebycheff(dimension=problem.number_of_objectives()),
                     neighbourhood_selection_probability=0.9,
                     max_number_of_replaced_solutions=2,
                     neighbor_size=20,
                     weight_files_path=str(_ensure_moead_weights()),
                     termination_criterion=term)
    raise ValueError(name)


def _feasible(solutions):
    """Keep demand-covering solutions only. The dominance-based algorithms
    handle the shortfall constraint internally, but MOEA/D's scalar
    aggregation ignores jMetal constraints, so we filter after the run (the
    lambda=1000 shortfall penalty inside ``buyer_cost`` already steers its
    search toward the feasible region)."""
    return [s for s in solutions if s.constraints[0] >= -1e-9]


def run(cfg: MarketConfig, runs: int = 5, evals: int = 15000, pop: int = 100,
        seed: int | None = 42):
    """Run every algorithm ``runs`` times; return superfronts + metrics dataframe.

    ``seed`` makes the (otherwise stochastic) evolutionary search reproducible by
    seeding Python's ``random`` and NumPy, which jMetalPy draws from. Pass
    ``seed=None`` to keep it nondeterministic.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    all_solutions = {n: [] for n in ALGORITHMS}
    for name in ALGORITHMS:
        for _ in range(runs):
            problem = EnergyMarketProblem(cfg)
            algo = _build(name, problem, pop, evals)
            algo.run()
            all_solutions[name].extend(
                get_non_dominated_solutions(_feasible(algo.result())))

    superfronts = {n: get_non_dominated_solutions(all_solutions[n]) for n in ALGORITHMS}

    # Common reference front = non-dominated union of every algorithm's superfront.
    combined = get_non_dominated_solutions(sum(superfronts.values(), []))
    reference_front = np.array([s.objectives for s in combined])

    # Single, COMMON reference point for the hypervolume so HV is comparable across
    # algorithms: the per-objective worst over every algorithm's points, pushed out
    # by a small margin so even the worst point contributes. (Using each front's own
    # max would make the HV values incomparable between algorithms.)
    all_points = np.vstack([np.array([s.objectives for s in superfronts[n]])
                            for n in ALGORITHMS])
    span = np.ptp(all_points, axis=0)
    hv_ref = list(np.amax(all_points, axis=0) + 0.01 * span)

    rows = []
    for name in ALGORITHMS:
        front = np.array([s.objectives for s in superfronts[name]])
        rows.append({
            "Algorithm": name,
            "Solutions": len(front),
            "HV": HyperVolume(reference_point=hv_ref).compute(front),
            "GD": GenerationalDistance(reference_front=reference_front).compute(front),
            "IGD": InvertedGenerationalDistance(reference_front=reference_front).compute(front),
            "Epsilon": EpsilonIndicator(reference_front=reference_front).compute(front),
            "Spread": spread(front, reference_front),
        })
    metrics = pd.DataFrame(rows)
    return superfronts, metrics, combined


def _front_to_df(front) -> pd.DataFrame:
    recs = []
    for s in front:
        ps, pw, cost = decode_objectives(s.objectives)
        q_solar, q_wind, p_solar, p_wind = s.variables
        recs.append({
            "Profit Solar (AS)": ps, "Profit Wind (AE)": pw, "Buyer Cost (AC)": cost,
            "q_solar": q_solar, "q_wind": q_wind, "p_solar": p_solar, "p_wind": p_wind,
        })
    return pd.DataFrame(recs)


def save_outputs(cfg: MarketConfig, superfronts, metrics, combined, suffix: str = ""):
    RESULTS.mkdir(exist_ok=True)
    metrics.to_csv(RESULTS / f"metrics_summary{suffix}.csv", index=False)
    for name, front in superfronts.items():
        _front_to_df(front).to_csv(RESULTS / f"superfront_{name}{suffix}.csv", index=False)
    _front_to_df(combined).to_csv(RESULTS / f"reference_front{suffix}.csv", index=False)

    # 2D projection: profit AS vs profit AE, colored by buyer cost.
    fig, ax = plt.subplots(figsize=(8, 6))
    df = _front_to_df(combined)
    sc = ax.scatter(df["Profit Solar (AS)"], df["Profit Wind (AE)"],
                    c=df["Buyer Cost (AC)"], cmap="coolwarm", s=35)
    ax.set_xlabel("Profit Solar (AS)")
    ax.set_ylabel("Profit Wind (AE)")
    ax.set_title(f"Pareto reference front — scenario {cfg.label}\n"
                 f"demand={cfg.demand:.1f}, gen_solar={cfg.gen_solar:.1f}, "
                 f"gen_wind={cfg.gen_wind:.1f}")
    fig.colorbar(sc, ax=ax, label="Buyer Cost (AC) — minimize")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / f"pareto_front{suffix}.png", dpi=130)
    plt.close(fig)

    # True 3D view of the 3-objective front (the 2D plot is a projection).
    fig = plt.figure(figsize=(7.5, 6))
    ax3 = fig.add_subplot(projection="3d")
    sc = ax3.scatter(df["Profit Solar (AS)"], df["Profit Wind (AE)"],
                     df["Buyer Cost (AC)"], c=df["Buyer Cost (AC)"],
                     cmap="coolwarm", s=25)
    ax3.set_xlabel("Profit Solar (AS)")
    ax3.set_ylabel("Profit Wind (AE)")
    ax3.set_zlabel("Buyer Cost (AC)")
    ax3.set_title(f"Pareto reference front (3D) — scenario {cfg.label}")
    fig.colorbar(sc, ax=ax3, shrink=0.6, label="Buyer Cost (AC) — minimize")
    fig.tight_layout()
    fig.savefig(RESULTS / f"pareto_front_3d{suffix}.png", dpi=130)
    plt.close(fig)

    # Parallel coordinates: one polyline per solution across the 3 objectives
    # (normalized per axis), the standard view for m >= 3 objectives.
    cols = ["Profit Solar (AS)", "Profit Wind (AE)", "Buyer Cost (AC)"]
    norm = (df[cols] - df[cols].min()) / (df[cols].max() - df[cols].min())
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("coolwarm")
    for _, row in norm.iterrows():
        ax.plot(range(len(cols)), row[cols], alpha=0.25,
                color=cmap(row["Buyer Cost (AC)"]))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["Profit AS\n(max)", "Profit AE\n(max)", "Cost AC\n(min)"])
    ax.set_ylabel("normalized objective value")
    ax.set_title(f"Reference front, parallel coordinates — scenario {cfg.label}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / f"pareto_parallel{suffix}.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--timestamp", type=str, help="Date present in both datasets")
    g.add_argument("--row", type=int, help="Row index in the datasets")
    ap.add_argument("--demand", type=float, default=None)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--evals", type=int, default=15000)
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for reproducibility (use -1 for nondeterministic)")
    ap.add_argument("--curved", action="store_true",
                    help="convex asymmetric generation costs + dispatch freedom → a "
                         "genuinely curved Pareto front (vs the mc=0 degenerate plane)")
    args = ap.parse_args()

    ts = args.timestamp if args.timestamp is not None else (args.row if args.row is not None else 4000)
    if args.curved:
        from src.optimization.market import scenario_curved
        cfg = scenario_curved(ts)
        suffix = "_curved"
    else:
        cfg = scenario_from_timestamp(ts, demand=args.demand)
        suffix = ""
    print(f"Scenario {cfg.label}{' [curved/econ-dispatch]' if args.curved else ''}: "
          f"demand={cfg.demand:.2f} gen_solar={cfg.gen_solar:.2f} gen_wind={cfg.gen_wind:.2f}")

    seed = None if args.seed == -1 else args.seed
    superfronts, metrics, combined = run(cfg, runs=args.runs, evals=args.evals,
                                         pop=args.pop, seed=seed)
    save_outputs(cfg, superfronts, metrics, combined, suffix=suffix)
    print(metrics.to_string(index=False))
    print(f"Reference front: {len(combined)} solutions. Outputs in {RESULTS}/")


if __name__ == "__main__":
    main()
