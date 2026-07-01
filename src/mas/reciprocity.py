"""Reciprocity in the repeated market — tit-for-tat dynamics (M2).

The Contract Net is a *repeated* game (several rounds per scenario), which is
exactly the setting where retributive strategies matter: the guia de SMA notes
that "if the opponent defects, it is punished the next round; the fear of
retaliation fosters cooperation". This script demonstrates the
:class:`~src.mas.strategies.ReciprocalStrategy` (tit-for-tat: nice, retaliatory,
forgiving) against three kinds of rival and shows the price trajectory round by
round:

  * vs **another reciprocator** -> mutual restraint holds the cooperative
    (high) price indefinitely: cooperation is sustained;
  * vs an **undercutter** (info_hiding) -> the reciprocator answers with a price
    war for a couple of rounds, then forgives (probes cooperation again); since
    the rival keeps undercutting, the punish/forgive cycle repeats — cooperation
    cannot be sustained unilaterally, it takes two reciprocators;
  * vs an **over-pricer** (opponent_modeling, which sits above the cooperative
    price) -> no retaliation: tit-for-tat punishes *undercutting*, not high
    prices.

This is the repeated-game counterpart of the one-shot Nash analysis
(:mod:`src.mas.game_analysis`): the static equilibrium is mutual exploitation,
whereas reciprocity is the mechanism that can hold the cooperative outcome.

Outputs (written to ``results/``):
  - ``mas_reciprocity.png``   one panel per matchup, price over rounds, price-war
                              rounds shaded
  - ``mas_reciprocity.csv``   per-round prices/profits/cost for every matchup

Usage:
    python -m src.mas.reciprocity [--timestamp "2017-06-15 09:00:00"] [--rounds 12]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.mas.simulation import run_negotiation
from src.optimization.market import scenario_from_timestamp

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# (solar_strategy, wind_strategy, short takeaway shown as the panel subtitle)
MATCHUPS = [
    ("tit_for_tat", "tit_for_tat",
     "vs another reciprocator:\ncooperation sustained"),
    ("tit_for_tat", "info_hiding",
     "vs an undercutter:\nretaliate (price war) then forgive"),
    ("tit_for_tat", "opponent_modeling",
     "vs an over-pricer:\nno retaliation (only undercuts are punished)"),
]

# Price at which the reciprocator wages its price war (mc + 0.5 clipped to the
# band -> the floor); used only to shade the punishment rounds in the figure.
WAR_EPS = 1.0


def _war_rounds(prices, p_min):
    """Indices where the (solar) reciprocator dropped to the price-war floor."""
    return [i for i, p in enumerate(prices) if p is not None and p <= p_min + WAR_EPS]


def run_reciprocity(cfg, rounds: int = 12):
    results = {}
    for solar, wind, _ in MATCHUPS:
        results[(solar, wind)] = run_negotiation(cfg, solar, wind, rounds=rounds)
    return results


def plot_reciprocity(cfg, results, rounds: int = 12):
    fig, axes = plt.subplots(1, len(MATCHUPS), figsize=(4.8 * len(MATCHUPS), 4.3),
                             sharey=True)
    for ax, (solar, wind, takeaway) in zip(axes, MATCHUPS):
        res = results[(solar, wind)]
        r = [x["round"] for x in res.rounds]
        ps = [x["price_solar"] for x in res.rounds]
        pw = [x["price_wind"] for x in res.rounds]

        # Shade the rounds the reciprocator (AS) spends in a price war.
        for i in _war_rounds(ps, cfg.price_min):
            ax.axvspan(i - 0.5, i + 0.5, color="tab:red", alpha=0.10, lw=0)

        ax.plot(r, ps, marker="o", color="tab:orange",
                label=f"AS price ({solar})")
        ax.plot(r, pw, marker="s", color="tab:blue",
                label=f"AE price ({wind})")
        ax.axhline(cfg.price_max, ls="--", color="grey", lw=0.8)
        ax.axhline(cfg.price_min, ls="--", color="grey", lw=0.8)
        ax.set_xlabel("round")
        ax.set_title(takeaway, fontsize=9)
        ax.legend(fontsize=7, loc="center right")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("asked unit price")
    fig.suptitle("Tit-for-tat reciprocity over rounds — AS = reciprocator "
                 f"(scenario {cfg.label}); shaded = price-war (punishment) rounds",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "mas_reciprocity.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def save_csv(results) -> Path:
    rows = []
    for (solar, wind), res in results.items():
        for x in res.rounds:
            rows.append({
                "solar_strategy": solar, "wind_strategy": wind,
                "round": x["round"], "price_solar": x["price_solar"],
                "price_wind": x["price_wind"], "profit_solar": x["profit_solar"],
                "profit_wind": x["profit_wind"], "cost": x["cost"],
                "shortfall": x["shortfall"],
            })
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "mas_reciprocity.csv"
    df.to_csv(out, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamp", type=str, default="2017-06-15 09:00:00",
                    help="a scenario where both sellers have comparable generation, "
                         "so undercutting genuinely shifts volume")
    ap.add_argument("--rounds", type=int, default=12)
    args = ap.parse_args()

    cfg = scenario_from_timestamp(args.timestamp)
    print(f"Scenario {cfg.label}: demand={cfg.demand:.2f} "
          f"solar={cfg.gen_solar:.2f} wind={cfg.gen_wind:.2f} kW\n")

    results = run_reciprocity(cfg, rounds=args.rounds)
    for (solar, wind), res in results.items():
        ps = [round(x["price_solar"], 1) for x in res.rounds]
        print(f"{solar}(AS) vs {wind}(AE)")
        print(f"  AS price/round: {ps}")
        print(f"  final  profit AS={res.profit_solar:.1f}  AE={res.profit_wind:.1f}  "
              f"buyer_cost={res.buyer_cost:.1f}  shortfall={res.shortfall:.3f}\n")

    fig_out = plot_reciprocity(cfg, results, rounds=args.rounds)
    csv_out = save_csv(results)
    print(f"Saved -> {fig_out}")
    print(f"Saved -> {csv_out}")


if __name__ == "__main__":
    main()
