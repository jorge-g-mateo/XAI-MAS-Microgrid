"""Statistical comparison of NSGA-II / NSGA-III / SPEA2 over many seeds.

A single run of a stochastic optimizer is anecdotal. Here we run each algorithm
over ``n_seeds`` independent seeds, compute four quality indicators per run
against a *common global reference front* (and a common hypervolume reference
point), and report mean +/- std plus boxplots. This is the standard way to claim
one algorithm is better than another on a multi-objective problem.

Outputs (written to ``results/``):
  - ``stats_runs.csv``        one row per (algorithm, seed): HV/GD/IGD/Epsilon
  - ``stats_summary.csv``     mean +/- std per algorithm
  - ``stats_boxplots.png``    2x2 boxplots, one per indicator

Usage:
    python -m src.optimization.stats --timestamp "2017-06-15 19:00:00" --seeds 15
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
from scipy.stats import friedmanchisquare, wilcoxon
from jmetal.core.quality_indicator import (
    EpsilonIndicator,
    GenerationalDistance,
    HyperVolume,
    InvertedGenerationalDistance,
)
from jmetal.util.solution import get_non_dominated_solutions

from src.optimization.market import scenario_from_timestamp
from src.optimization.metrics import spread
from src.optimization.problem import EnergyMarketProblem
from src.optimization.run_optimization import ALGORITHMS, _build, _feasible

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
INDICATORS = ["HV", "GD", "IGD", "Epsilon", "Spread"]


def run_stats(cfg, n_seeds: int = 15, evals: int = 10000, pop: int = 100,
              base_seed: int = 100):
    """Run each algorithm once per seed; return per-run and summary dataframes."""
    seeds = [base_seed + i for i in range(n_seeds)]
    fronts: dict[tuple[str, int], np.ndarray] = {}
    all_solutions = []

    for s in seeds:
        random.seed(s)
        np.random.seed(s)
        for name in ALGORITHMS:
            problem = EnergyMarketProblem(cfg)
            algo = _build(name, problem, pop, evals)
            algo.run()
            nd = get_non_dominated_solutions(_feasible(algo.result()))
            fronts[(name, s)] = np.array([sol.objectives for sol in nd])
            all_solutions.extend(nd)

    # Common global reference front + common HV reference point.
    combined = get_non_dominated_solutions(all_solutions)
    reference_front = np.array([sol.objectives for sol in combined])
    all_pts = np.vstack([f for f in fronts.values()])
    span = np.ptp(all_pts, axis=0)
    hv_ref = list(np.amax(all_pts, axis=0) + 0.01 * span)

    rows = []
    for (name, s), front in fronts.items():
        rows.append({
            "Algorithm": name, "seed": s,
            "HV": HyperVolume(reference_point=hv_ref).compute(front),
            "GD": GenerationalDistance(reference_front=reference_front).compute(front),
            "IGD": InvertedGenerationalDistance(reference_front=reference_front).compute(front),
            "Epsilon": EpsilonIndicator(reference_front=reference_front).compute(front),
            "Spread": spread(front, reference_front),
        })
    runs = pd.DataFrame(rows)

    summary = runs.groupby("Algorithm")[INDICATORS].agg(["mean", "std"]).round(4)

    # Per-algorithm non-dominated union across seeds (for the overlay figure).
    union = {}
    for name in ALGORITHMS:
        pts = [fronts[(name, s)] for s in seeds if (name, s) in fronts]
        union[name] = np.vstack(pts) if pts else np.empty((0, 3))
    return runs, summary, len(combined), union


def _front_overlay(union: dict[str, np.ndarray], suffix: str = ""):
    """Overlay the three algorithms' fronts in objective space (profit AS vs AE).
    Objectives are stored as [-profit_s, -profit_w, cost]; negate to read profits."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = {"NSGAII": "tab:blue", "NSGAIII": "tab:green", "SPEA2": "tab:red",
              "MOEAD": "tab:purple"}
    markers = {"NSGAII": "o", "NSGAIII": "^", "SPEA2": "s", "MOEAD": "D"}
    for name, pts in union.items():
        if len(pts) == 0:
            continue
        ax.scatter(-pts[:, 0], -pts[:, 1], s=14, alpha=0.5,
                   c=colors.get(name), marker=markers.get(name, "o"), label=name)
    ax.set_xlabel("Profit Solar (AS)"); ax.set_ylabel("Profit Wind (AE)")
    ax.set_title("Pareto fronts by algorithm (profit AS vs. AE)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / f"stats_front_overlay{suffix}.png", dpi=130)
    plt.close(fig)


def significance_tests(runs: pd.DataFrame) -> pd.DataFrame:
    """Friedman omnibus test across the three algorithms (paired by seed) per
    indicator, plus pairwise Wilcoxon signed-rank with Holm correction.

    A small Friedman p-value means the algorithms are not all equivalent on that
    indicator; the Holm-corrected pairwise p-values then say which pairs differ.
    """
    rows = []
    for ind in INDICATORS:
        per_algo = {a: runs[runs.Algorithm == a].sort_values("seed")[ind].to_numpy()
                    for a in ALGORITHMS}
        chi, p_fried = friedmanchisquare(*[per_algo[a] for a in ALGORITHMS])

        # pairwise Wilcoxon, Holm-corrected over all pairs of algorithms
        from itertools import combinations
        pairs = list(combinations(ALGORITHMS, 2))
        raw = []
        for a, b in pairs:
            try:
                _, p = wilcoxon(per_algo[a], per_algo[b])
            except ValueError:      # identical vectors -> no difference
                p = 1.0
            raw.append(p)
        order = np.argsort(raw)
        holm = [0.0] * len(raw)
        for rank, i in enumerate(order):
            holm[i] = min(1.0, raw[i] * (len(raw) - rank))
        holm = np.maximum.accumulate([holm[i] for i in order])
        holm_by_pair = {pairs[order[k]]: holm[k] for k in range(len(pairs))}

        rows.append({
            "indicator": ind,
            "friedman_chi2": round(chi, 3), "friedman_p": round(p_fried, 4),
            **{f"p_{a}_vs_{b}": round(holm_by_pair[(a, b)], 4) for a, b in pairs},
        })
    return pd.DataFrame(rows)


def save_outputs(runs: pd.DataFrame, summary: pd.DataFrame,
                 union: dict[str, np.ndarray], suffix: str = ""):
    RESULTS.mkdir(exist_ok=True)
    runs.to_csv(RESULTS / f"stats_runs{suffix}.csv", index=False)
    summary.to_csv(RESULTS / f"stats_summary{suffix}.csv")
    _front_overlay(union, suffix=suffix)

    ncols = 3
    nrows = -(-len(INDICATORS) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(INDICATORS):]:
        ax.set_visible(False)
    for ax, ind in zip(axes, INDICATORS):
        data = [runs[runs.Algorithm == a][ind].to_numpy() for a in ALGORITHMS]
        ax.boxplot(data, tick_labels=ALGORITHMS, showmeans=True)
        ax.set_title(ind)
        ax.grid(True, axis="y", alpha=0.3)
        if ind != "HV":
            ax.set_ylabel("lower is better")
        else:
            ax.set_ylabel("higher is better")
    fig.suptitle("Quality-indicator distributions over independent seeds")
    fig.tight_layout()
    fig.savefig(RESULTS / f"stats_boxplots{suffix}.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--timestamp", type=str)
    g.add_argument("--row", type=int)
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--evals", type=int, default=10000)
    ap.add_argument("--curved", action="store_true",
                    help="convex-cost / dispatch-freedom scenario (curved front) "
                         "instead of the mc=0 degenerate plane")
    ap.add_argument("--from-csv", action="store_true",
                    help="skip optimization; recompute the significance tests from "
                         "an existing results/stats_runs.csv")
    args = ap.parse_args()

    pd.set_option("display.width", 200)
    suffix = "_curved" if args.curved else ""

    if args.from_csv:
        runs = pd.read_csv(RESULTS / f"stats_runs{suffix}.csv")
        sig = significance_tests(runs)
        sig.to_csv(RESULTS / f"stats_significance{suffix}.csv", index=False)
        print("Significance tests (Friedman + Holm-corrected pairwise Wilcoxon):\n")
        print(sig.to_string(index=False))
        return

    ts = args.timestamp if args.timestamp is not None else (
        args.row if args.row is not None else "2017-06-15 19:00:00")
    if args.curved:
        from src.optimization.market import scenario_curved
        cfg = scenario_curved(ts)
    else:
        cfg = scenario_from_timestamp(ts)
    print(f"Scenario {cfg.label}{' [curved]' if args.curved else ''}: demand={cfg.demand:.2f} "
          f"solar={cfg.gen_solar:.2f} wind={cfg.gen_wind:.2f} kW")
    print(f"Running {args.seeds} seeds x {len(ALGORITHMS)} algorithms...")

    runs, summary, n_ref, union = run_stats(cfg, n_seeds=args.seeds, evals=args.evals)
    save_outputs(runs, summary, union, suffix=suffix)
    sig = significance_tests(runs)
    sig.to_csv(RESULTS / f"stats_significance{suffix}.csv", index=False)

    print(f"\nGlobal reference front: {n_ref} solutions.\n")
    print(summary.to_string())
    print("\nSignificance (Friedman + Holm-corrected Wilcoxon):")
    print(sig.to_string(index=False))
    print(f"\nOutputs -> {RESULTS}/ (stats_*.{{csv,png}})")


if __name__ == "__main__":
    main()
