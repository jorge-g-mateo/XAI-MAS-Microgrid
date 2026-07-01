"""Optimization across scenarios: how the Pareto front shifts with generation.

The single-scenario optimization shows one operating point; this scans a grid of
scenarios (times of day x seasons) and shows how the optimal market changes as the
available solar/wind generation changes. It reports, per scenario, the minimum
achievable consumer cost and the maximum total seller profit on the Pareto front,
and overlays a few representative fronts.

Outputs (results/):
  - opt_scenarios.csv         per-scenario summary (gen, demand, min cost, max profit)
  - opt_scenarios.png         min cost vs. generation + overlaid fronts

Usage:
    python -m src.optimization.scenario_scan --runs 2 --evals 6000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.getLogger("jmetal").setLevel(logging.WARNING)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.optimization.market import scenario_from_timestamp
from src.optimization.problem import decode_objectives
from src.optimization.run_optimization import run as run_opt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# Day/night across winter and summer (kept small to bound runtime).
HOURS = [3, 9, 13, 19]
MONTHS = [1, 7]
YEAR = 2018


def _select(year=YEAR, hours=HOURS, months=MONTHS) -> list[str]:
    from src.common.inference import add_date_features, load_solar_data
    df = add_date_features(load_solar_data())
    df["_dt"] = pd.to_datetime(df["Date"])
    df = df[df["_dt"].dt.year == year]
    out = []
    for m in months:
        for h in hours:
            match = df[(df["_dt"].dt.month == m) & (df["hour"] == h)]
            if not match.empty:
                out.append(str(match.iloc[0]["Date"]))
    return out


def scan(runs: int = 2, evals: int = 6000):
    scenarios = _select()
    summary, fronts = [], {}
    for i, ts in enumerate(scenarios, 1):
        cfg = scenario_from_timestamp(ts)
        _, _, combined = run_opt(cfg, runs=runs, evals=evals, seed=42)
        front = np.array([decode_objectives(s.objectives) for s in combined])  # ps,pw,cost
        fronts[ts] = front
        summary.append({
            "scenario": ts, "hour": pd.to_datetime(ts).hour,
            "gen_solar": round(cfg.gen_solar, 3), "gen_wind": round(cfg.gen_wind, 3),
            "gen_total": round(cfg.gen_solar + cfg.gen_wind, 3),
            "demand": round(cfg.demand, 3),
            "min_buyer_cost": round(float(front[:, 2].min()), 2),
            "max_total_profit": round(float((front[:, 0] + front[:, 1]).max()), 2),
        })
        print(f"[{i}/{len(scenarios)}] {ts}: gen_total={cfg.gen_solar+cfg.gen_wind:.2f} "
              f"min_cost={front[:,2].min():.1f}")
    return pd.DataFrame(summary), fronts


def save_outputs(summary: pd.DataFrame, fronts: dict):
    RESULTS.mkdir(exist_ok=True)
    summary.to_csv(RESULTS / "opt_scenarios.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Panel A: minimum achievable consumer cost vs available generation.
    sc = ax1.scatter(summary["gen_total"], summary["min_buyer_cost"],
                     c=summary["hour"], cmap="twilight", s=60)
    ax1.set_xlabel("total available generation (kW)")
    ax1.set_ylabel("minimum consumer cost on the front")
    ax1.set_title("Optimal cost vs. generation")
    ax1.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax1, label="hour of day")

    # Panel B: overlay the profit fronts of three representative scenarios.
    reps = summary.sort_values("gen_total").iloc[[0, len(summary) // 2, -1]]
    for _, row in reps.iterrows():
        f = fronts[row["scenario"]]
        ax2.scatter(f[:, 0], f[:, 1], s=12, alpha=0.5,
                    label=f"{row['scenario'][11:16]} (gen {row['gen_total']:.1f})")
    ax2.set_xlabel("Profit Solar (AS)"); ax2.set_ylabel("Profit Wind (AE)")
    ax2.set_title("Pareto fronts across scenarios")
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

    fig.suptitle("Optimization across day/night and seasons")
    fig.tight_layout()
    fig.savefig(RESULTS / "opt_scenarios.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--evals", type=int, default=6000)
    args = ap.parse_args()

    summary, fronts = scan(runs=args.runs, evals=args.evals)
    save_outputs(summary, fronts)
    pd.set_option("display.width", 200)
    print("\n" + summary.to_string(index=False))
    print(f"\nOutputs -> {RESULTS}/ (opt_scenarios.{{csv,png}})")


if __name__ == "__main__":
    main()
