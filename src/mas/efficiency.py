"""Efficiency of the negotiation: price of anarchy and scarcity sensitivity.

Two analyses that quantify *how costly* self-interested negotiation is for the
consumer relative to the ideal, and how that cost depends on resource scarcity.

Price of anarchy (PoA)
----------------------
The socially optimal outcome for the consumer is to be served at the competitive
price floor: ``optimal = served*price_min + shortfall*penalty``. The PoA of a
strategy profile is its negotiated buyer cost divided by this optimum
(PoA = 1 means perfectly efficient; larger is worse).

Scarcity sweep
--------------
Demand is swept from abundant to scarce (as a ratio of total available
generation). For a few representative strategy profiles we track buyer cost,
shortfall and PoA, exposing the demand regime in which each strategy starts to
hurt the consumer.

Outputs (results/):
  - mas_poa.csv               PoA for every strategy profile (evening scenario)
  - mas_scarcity.csv          sweep rows
  - mas_scarcity.png          buyer cost / shortfall / PoA vs. demand ratio

Usage:
    python -m src.mas.efficiency --timestamp "2017-06-15 19:00:00"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.mas.simulation import run_negotiation
from src.mas.strategies import STRATEGIES
from src.optimization.market import MarketConfig, scenario_from_timestamp

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# Representative profiles for the scarcity sweep.
PROFILES = [("honest", "honest"), ("opponent_modeling", "opponent_modeling"),
            ("info_hiding", "info_hiding"), ("deception", "deception")]


def optimal_buyer_cost(cfg: MarketConfig) -> float:
    """Socially optimal (cheapest) cost to serve the consumer: all served energy at
    the competitive price floor, plus the unavoidable shortfall penalty."""
    gen_total = cfg.gen_solar + cfg.gen_wind
    served = min(cfg.demand, gen_total)
    shortfall = max(0.0, cfg.demand - gen_total)
    return served * cfg.price_min + shortfall * cfg.shortfall_penalty


def poa_table(cfg: MarketConfig, rounds: int = 12) -> pd.DataFrame:
    """Price of anarchy for every (solar, wind) strategy profile."""
    opt = optimal_buyer_cost(cfg)
    names = list(STRATEGIES)
    rows = []
    for s in names:
        for w in names:
            res = run_negotiation(cfg, s, w, rounds=rounds)
            rows.append({"solar_strategy": s, "wind_strategy": w,
                         "buyer_cost": round(res.buyer_cost, 2),
                         "optimal_cost": round(opt, 2),
                         "price_of_anarchy": round(res.buyer_cost / opt, 3),
                         "shortfall": round(res.shortfall, 3)})
    return pd.DataFrame(rows)


def scarcity_sweep(timestamp, ratios=None, profiles=PROFILES,
                   rounds: int = 12) -> pd.DataFrame:
    """Sweep demand from abundant to scarce; track cost/shortfall/PoA per profile."""
    ratios = ratios if ratios is not None else np.round(np.arange(0.4, 1.31, 0.1), 2)
    rows = []
    for ratio in ratios:
        cfg = scenario_from_timestamp(timestamp, demand_ratio=float(ratio))
        opt = optimal_buyer_cost(cfg)
        for s, w in profiles:
            res = run_negotiation(cfg, s, w, rounds=rounds)
            rows.append({"demand_ratio": float(ratio), "profile": f"{s}/{w}",
                         "buyer_cost": res.buyer_cost, "shortfall": res.shortfall,
                         "price_of_anarchy": res.buyer_cost / opt})
    return pd.DataFrame(rows)


def _plot_scarcity(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = [("buyer_cost", "buyer cost"), ("shortfall", "shortfall (kW)"),
               ("price_of_anarchy", "price of anarchy")]
    for ax, (col, ylabel) in zip(axes, metrics):
        for prof, g in df.groupby("profile"):
            ax.plot(g["demand_ratio"], g[col], marker=".", label=prof)
        ax.set_xlabel("demand / total generation")
        ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3)
        if col == "price_of_anarchy":
            ax.axhline(1.0, color="grey", ls="--", lw=0.8)
    axes[0].set_title("Consumer cost vs. scarcity")
    axes[1].set_title("Shortfall vs. scarcity")
    axes[2].set_title("Efficiency loss (PoA) vs. scarcity")
    axes[2].legend(fontsize=7)
    fig.suptitle("Negotiation efficiency as resources become scarce")
    fig.tight_layout()
    fig.savefig(RESULTS / "mas_scarcity.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamp", type=str, default="2017-06-15 19:00:00")
    ap.add_argument("--rounds", type=int, default=12)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    cfg = scenario_from_timestamp(args.timestamp)
    print(f"Scenario {cfg.label}: demand={cfg.demand:.2f} "
          f"solar={cfg.gen_solar:.2f} wind={cfg.gen_wind:.2f} kW")
    print(f"Optimal (competitive) buyer cost: {optimal_buyer_cost(cfg):.2f}\n")

    poa = poa_table(cfg, rounds=args.rounds)
    poa.to_csv(RESULTS / "mas_poa.csv", index=False)
    pd.set_option("display.width", 200)
    print("Price of anarchy by strategy profile (sorted):")
    print(poa.sort_values("price_of_anarchy").to_string(index=False))

    sweep = scarcity_sweep(args.timestamp, rounds=args.rounds)
    sweep.round(3).to_csv(RESULTS / "mas_scarcity.csv", index=False)
    _plot_scarcity(sweep)
    print(f"\nOutputs -> {RESULTS}/ (mas_poa.csv, mas_scarcity.{{csv,png}})")


if __name__ == "__main__":
    main()
