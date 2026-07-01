"""Consumer-side bargaining dynamics (M8) — the concession trajectory.

The headline gap in the previous write-up was that *the consumer's negotiation
never appeared*: earlier experiments used the ``price_taker`` buyer, so only
the sellers moved. This script makes the **bilateral** concession explicit: in
each round the buyer opens low and raises while the seller concedes down, and
they meet at the cleared price. We plot that trajectory (buyer price up, seller
price down) for the four consumer strategies against surplus-extracting
(opponent_modeling) sellers, where the bargaining gap is widest.

Outputs (``results/``):
  - ``mas_consumer_concession.png``  buyer vs seller price across bargaining steps,
                                     one panel per consumer strategy

Usage:
    python -m src.mas.consumer_dynamics --timestamp "2017-06-15 19:00:00"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.mas.buyer_strategies import BUYER_STRATEGIES
from src.mas.simulation import SOLAR, run_negotiation
from src.optimization.market import scenario_from_timestamp

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def plot_concession(cfg, sellers=("opponent_modeling", "opponent_modeling"),
                    rounds: int = 8, n_bargain: int = 4):
    buyers = list(BUYER_STRATEGIES)
    fig, axes = plt.subplots(1, len(buyers), figsize=(4 * len(buyers), 3.8),
                             sharey=True)
    for ax, buyer in zip(axes, buyers):
        res = run_negotiation(cfg, sellers[0], sellers[1], buyer_strategy=buyer,
                              rounds=rounds, n_bargain=n_bargain)
        log = pd.DataFrame(res.bargain_log)
        # Use the last round (sellers have climbed → widest bargaining gap) and the
        # solar seller's thread as the representative concession trajectory.
        last = log[(log["round"] == log["round"].max()) & (log["seller"] == SOLAR)]
        ax.plot(last["step"], last["buyer_price"], marker="o", color="tab:green",
                label="buyer counter (AC)")
        ax.plot(last["step"], last["seller_price"], marker="s", color="tab:orange",
                label="seller ask (AS)")
        ax.axhline(cfg.price_min, ls="--", color="grey", lw=0.8)
        ax.axhline(cfg.price_max, ls="--", color="grey", lw=0.8)
        ax.set_title(f"{buyer}\ncost={res.buyer_cost:.0f}  short={res.shortfall:.2f}",
                     fontsize=9)
        ax.set_xlabel("bargaining step")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("unit price")
    axes[0].legend(fontsize=8, loc="center right")
    fig.suptitle("Consumer concession vs. opponent-modeling sellers: the buyer raises, "
                 "the seller concedes, they meet (last round)")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "mas_consumer_concession.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamp", type=str, default="2017-06-15 19:00:00")
    ap.add_argument("--solar", type=str, default="opponent_modeling")
    ap.add_argument("--wind", type=str, default="opponent_modeling")
    args = ap.parse_args()

    cfg = scenario_from_timestamp(args.timestamp)
    out = plot_concession(cfg, sellers=(args.solar, args.wind))
    print(f"Scenario {cfg.label}: demand={cfg.demand:.2f} "
          f"solar={cfg.gen_solar:.2f} wind={cfg.gen_wind:.2f} kW")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
