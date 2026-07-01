"""Impact of a negotiating consumer (M8/M9).

The original model had a price-taking consumer, so sellers saturated the price at
``p_max``. Here we make the consumer bargain (counter-offers + reservation price)
and measure how each buyer strategy moves the outcome, against two seller
backdrops (both honest vs. both opponent-modeling). It reports clearing price,
consumer cost, shortfall and seller profits per (buyer × sellers) combination.

Key reading:
  * ``price_taker``            reproduces the old saturation (baseline).
  * ``opponent_modeling_buyer`` is the rational buyer: lowest cost *without*
                               shortfall (extracts surplus, keeps supply).
  * ``hard_bargainer``          can over-bargain → shortfall (the cautionary case).

Outputs (results/):
  - mas_buyer_comparison.csv
  - mas_buyer_comparison.png   consumer cost + clearing price per buyer strategy

Usage:
    python -m src.mas.buyer_experiment --timestamp "2017-06-15 19:00:00"
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
from src.mas.simulation import run_negotiation
from src.optimization.market import scenario_from_timestamp

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

SELLER_BACKDROPS = [("honest", "honest"), ("opponent_modeling", "opponent_modeling")]
BUYERS = list(BUYER_STRATEGIES)  # price_taker, honest_buyer, hard_bargainer, opponent_modeling_buyer


def run(timestamp: str) -> pd.DataFrame:
    cfg = scenario_from_timestamp(timestamp)
    rows = []
    for ss, ws in SELLER_BACKDROPS:
        for buyer in BUYERS:
            r = run_negotiation(cfg, ss, ws, buyer_strategy=buyer)
            rows.append({
                "sellers": f"{ss}/{ws}", "buyer": buyer,
                "clearing_price": round((r.final_price_solar + r.final_price_wind) / 2, 2),
                "consumer_cost": round(r.buyer_cost, 2),
                "shortfall": round(r.shortfall, 3),
                "profit_solar": round(r.profit_solar, 2),
                "profit_wind": round(r.profit_wind, 2),
            })
    return pd.DataFrame(rows)


def save_outputs(df: pd.DataFrame, timestamp: str):
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "mas_buyer_comparison.csv", index=False)

    backdrops = df["sellers"].unique()
    x = np.arange(len(BUYERS))
    width = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    for i, bd in enumerate(backdrops):
        sub = df[df["sellers"] == bd].set_index("buyer").reindex(BUYERS)
        ax1.bar(x + (i - 0.5) * width, sub["consumer_cost"], width, label=f"sellers: {bd}")
        ax2.bar(x + (i - 0.5) * width, sub["clearing_price"], width, label=f"sellers: {bd}")
        # mark shortfall cases on the cost panel
        for xi, (cost, sf) in enumerate(zip(sub["consumer_cost"], sub["shortfall"])):
            if sf > 1e-3:
                ax1.text(xi + (i - 0.5) * width, cost, "shortfall!", rotation=90,
                         ha="center", va="bottom", fontsize=7, color="red")

    for ax, title, ylab in [(ax1, "Consumer cost by buyer strategy", "consumer cost"),
                            (ax2, "Clearing price by buyer strategy", "mean clearing price")]:
        ax.set_xticks(x); ax.set_xticklabels(BUYERS, rotation=20, ha="right", fontsize=8)
        ax.set_title(title); ax.set_ylabel(ylab); ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    ax2.axhline(80, color="grey", ls=":", lw=1)
    ax2.text(0, 80.3, "price ceiling (p_max)", fontsize=7, color="grey")

    fig.suptitle(f"A negotiating consumer breaks the price saturation — {timestamp}")
    fig.tight_layout()
    fig.savefig(RESULTS / "mas_buyer_comparison.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamp", default="2017-06-15 19:00:00")
    args = ap.parse_args()

    df = run(args.timestamp)
    save_outputs(df, args.timestamp)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"\nOutputs -> {RESULTS}/ (mas_buyer_comparison.{{csv,png}})")


if __name__ == "__main__":
    main()
