"""Unified comparison: emergent negotiation vs. the optimal Pareto front (D).

The optimization and the MAS share one market definition, so a negotiated outcome
can be placed on the *same* objective axes as the Pareto front and we can ask, for
**every** strategy pairing and **every** consumer strategy at once, how far emergent
negotiation lands from the cost-aware optimum. We run this on the **curved**
(convex-cost, dispatch-freedom) scenario so the front is a genuine trade-off
surface (not the degenerate plane), and the MAS profits are scored with the same
convex cost model (see :func:`run_negotiation`), making the two directly comparable.

Outputs (``results/``):
  - ``mas_opt_comparison.csv``   the 6x6 x {4 buyers} grid scored vs the curved front
  - ``mas_opt_comparison.png``   negotiated points over the curved front + per-buyer
                                 mean distance to the front

Usage:
    python -m src.mas.opt_comparison --timestamp "2017-06-15 13:00:00"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.mas.buyer_strategies import BUYER_STRATEGIES
from src.mas.run_mas import _compare_to_front, _grid
from src.optimization.market import scenario_curved
from src.optimization.problem import decode_objectives
from src.optimization.run_optimization import run as run_opt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def unified(cfg, rounds: int = 12, runs: int = 3, evals: int = 8000):
    """Curved Pareto front + the 6x6 x {4 buyers} grid scored against it."""
    _, _, combined = run_opt(cfg, runs=runs, evals=evals)
    front = np.array([decode_objectives(s.objectives) for s in combined])  # ps, pw, cost

    parts = []
    for b in BUYER_STRATEGIES:
        parts.append(_compare_to_front(_grid(cfg, rounds, b), front))
    grid = pd.concat(parts, ignore_index=True)
    return front, grid


def _extractive(row) -> bool:
    return ("info_hiding" in (row.solar_strategy, row.wind_strategy)
            or "deception" in (row.solar_strategy, row.wind_strategy))


def save_outputs(cfg, front, grid):
    RESULTS.mkdir(exist_ok=True)
    grid.to_csv(RESULTS / "mas_opt_comparison.csv", index=False)

    buyers = list(BUYER_STRATEGIES)
    colors = dict(zip(buyers, ["tab:blue", "tab:green", "tab:red", "tab:purple"]))
    # Separate served outcomes (a real allocation to place on the front) from
    # walk-aways: an over-aggressive consumer can refuse every offer and go
    # unserved, which the shortfall penalty sends far off the front — a distinct
    # phenomenon (lost supply), not "distance to the optimum".
    served = grid[grid.shortfall < 1e-3]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   gridspec_kw={"width_ratios": [1.6, 1]})

    # Panel A: served negotiated points over the curved front (profit_solar vs cost).
    axA.scatter(front[:, 0], front[:, 2], s=12, color="lightgrey",
                label="curved Pareto front", zorder=1)
    for b in buyers:
        g = served[served.buyer_strategy == b]
        axA.scatter(g.profit_solar, g.buyer_cost, s=24, color=colors[b],
                    alpha=0.75, label=f"buyer={b}", zorder=2)
    for (ss, ws, mk, lbl) in [("honest", "honest", "*", "honest/honest"),
                              ("opponent_modeling", "opponent_modeling", "X", "opp/opp")]:
        sub = served[(served.solar_strategy == ss) & (served.wind_strategy == ws)
                     & (served.buyer_strategy == "honest_buyer")]
        if not sub.empty:
            r = sub.iloc[0]
            axA.scatter(r.profit_solar, r.buyer_cost, marker=mk, s=240,
                        facecolor="none", edgecolor="black", linewidth=1.8, zorder=3,
                        label=f"{lbl} (honest buyer)")
    cmin, cmax = front[:, 2].min(), front[:, 2].max()
    axA.set_ylim(cmin - 0.08 * (cmax - cmin), cmax + 0.25 * (cmax - cmin))
    axA.set_xlabel("Profit Solar (AS)")
    axA.set_ylabel("Buyer Cost (AC) — minimize")
    axA.set_title("Served negotiated outcomes vs. the curved Pareto front")
    axA.legend(fontsize=7, loc="upper left"); axA.grid(True, alpha=0.3)

    # Panel B: median distance-to-front over served outcomes (robust to walk-aways)
    # + the share of pairings that end in a shortfall, annotated per consumer.
    med = served.groupby("buyer_strategy")["dist_to_pareto"].median().reindex(buyers).fillna(0)
    short_rate = (grid.assign(sf=grid.shortfall > 1e-3)
                  .groupby("buyer_strategy")["sf"].mean().reindex(buyers) * 100)
    bars = axB.bar(range(len(buyers)), med.values, color=[colors[b] for b in buyers])
    for i, b in enumerate(buyers):
        axB.text(i, med.values[i], f"shortfall\n{short_rate[b]:.0f}%",
                 ha="center", va="bottom", fontsize=7)
    axB.set_xticks(range(len(buyers)))
    axB.set_xticklabels(buyers, rotation=25, ha="right", fontsize=8)
    axB.set_ylabel("median normalized distance to front (served)")
    axB.set_title("Closeness to the optimum by consumer strategy")
    axB.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Optimization ↔ negotiation, curved scenario {cfg.label} "
                 f"(D={cfg.demand:.2f}, gen_s={cfg.gen_solar:.2f}, gen_w={cfg.gen_wind:.2f})")
    fig.tight_layout()
    fig.savefig(RESULTS / "mas_opt_comparison.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamp", default="2017-06-15 13:00:00")
    ap.add_argument("--rounds", type=int, default=12)
    args = ap.parse_args()

    cfg = scenario_curved(args.timestamp)
    print(f"Curved scenario {cfg.label}: demand={cfg.demand:.2f} "
          f"gen_s={cfg.gen_solar:.2f} gen_w={cfg.gen_wind:.2f}")
    front, grid = unified(cfg, rounds=args.rounds)
    save_outputs(cfg, front, grid)

    pd.set_option("display.width", 200)
    served = grid[grid.shortfall < 1e-3]
    print(f"\nFront: {len(front)} points. Grid: {len(grid)} negotiated outcomes "
          f"({len(served)} served, {len(grid) - len(served)} ended in shortfall).")
    print("\nMedian distance to the curved front over SERVED outcomes "
          "(lower = closer to optimum):")
    print(served.groupby("buyer_strategy")["dist_to_pareto"].median().round(3).to_string())
    print("\n% of pairings that end in a shortfall (consumer walks away), by consumer strategy:")
    print(((grid.shortfall > 1e-3).groupby(grid.buyer_strategy).mean() * 100)
          .round(1).to_string())
    print(f"\nOutputs -> {RESULTS}/ (mas_opt_comparison.{{csv,png}})")


if __name__ == "__main__":
    main()
