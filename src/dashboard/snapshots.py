"""jMetal observer that captures Pareto front snapshots during optimization.

Used by the animated dashboard to replay the evolution of the algorithm
generation by generation.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from jmetal.util.solution import get_non_dominated_solutions

from src.optimization.market import MarketConfig
from src.optimization.problem import EnergyMarketProblem, decode_objectives
from src.optimization.run_optimization import _build


class SnapshotObserver:
    """Captures the non-dominated front every ``every_n`` evaluations.

    jMetal calls ``observer.update(**observable_data)`` after each generation.
    Observable keys: EVALUATIONS, SOLUTIONS, COMPUTING_TIME, PROBLEM.
    """

    def __init__(self, every_n: int = 200):
        self.every_n = every_n
        self.snapshots: list[dict] = []
        self._last_captured: int = -every_n  # capture generation 0 too

    def update(self, *args, **kwargs):
        evals = kwargs.get("EVALUATIONS", 0)
        solutions = kwargs.get("SOLUTIONS", [])
        if not solutions:
            return
        if evals - self._last_captured < self.every_n:
            return
        self._last_captured = evals
        non_dom = get_non_dominated_solutions(solutions)
        front = [list(decode_objectives(s.objectives)) for s in non_dom]
        self.snapshots.append({
            "evals": evals,
            "n_solutions": len(front),
            "front": front,  # list of [profit_solar, profit_wind, buyer_cost]
        })


def run_with_snapshots(
    cfg: MarketConfig,
    algorithm: str = "NSGAII",
    evals: int = 6000,
    pop: int = 100,
    every_n: int = 200,
    seed: int = 42,
) -> tuple[list[dict], np.ndarray]:
    """Run one algorithm and collect front snapshots for animation.

    Returns
    -------
    snapshots : list[dict]
        Each entry: {evals, n_solutions, front: [[ps, pw, cost], ...]}.
    final_front : np.ndarray  shape (N, 3)
        Final non-dominated front [profit_solar, profit_wind, buyer_cost].
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    problem = EnergyMarketProblem(cfg)
    algo = _build(algorithm, problem, pop, evals)

    observer = SnapshotObserver(every_n=every_n)
    algo.observable.register(observer=observer)
    algo.run()

    non_dom = get_non_dominated_solutions(algo.result())
    final_front = np.array([decode_objectives(s.objectives) for s in non_dom])

    # Always include the final state as the last snapshot.
    if not observer.snapshots or observer.snapshots[-1]["evals"] < evals:
        observer.snapshots.append({
            "evals": evals,
            "n_solutions": len(final_front),
            "front": final_front.tolist(),
        })

    return observer.snapshots, final_front
