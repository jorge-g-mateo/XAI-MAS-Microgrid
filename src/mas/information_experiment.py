"""Value of information: feedback richness vs enforcement of cooperation (M6).

A competitive-environment question the guia de SMA raises is *information hiding*:
what is the **value of the information** the market reveals? We ablate the
richness of the consumer's round feedback and measure the impact. The
:class:`~src.mas.agents.ConsumerAgent` supports three levels (``info_level``):

  * ``full``     — clearing price **and** each seller's best competitor price
    (``rival_price``, added in M5);
  * ``clearing`` — clearing price only (no ``rival_price``): the pre-M5 state;
  * ``blind``    — no prices at all (sellers learn only whether/how much they sold).

The sharpest effect is on **enforcement**: a reciprocal (tit-for-tat) seller
punishes a rival that breaks cooperation only if it can *detect* the breach.
With ``rival_price`` it detects even a **partial** undercut (it still sells its
residual, so a quantity-only signal would miss it); without it, it is blind to
anything short of being fully shut out — so an undercutter/deceiver exploits it
freely. Information is what makes the retaliation threat credible.

We therefore pit a tit-for-tat solar seller against the two exploitative rivals
(info_hiding, deception) at a *symmetric* scenario (so undercutting is partial,
the case the quantity signal cannot catch) and sweep the information level.

Outputs (results/):
  - ``mas_information.png``    price trajectory (full vs blind) + reciprocator
                               profit and consumer cost by information level
  - ``mas_information.csv``    the per-(rival, level) outcomes

Usage:
    python -m src.mas.information_experiment [--timestamp "2017-06-15 09:00:00"]
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
from src.optimization.market import scenario_from_timestamp

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

RECIPROCATOR = "tit_for_tat"        # solar seller (the enforcer)
RIVALS = ["info_hiding", "deception"]
LEVELS = ["full", "clearing", "blind"]
LEVEL_COLORS = {"full": "tab:green", "clearing": "tab:orange", "blind": "tab:red"}


def run_grid(cfg, rounds: int = 12):
    rows, traj = [], {}
    for rival in RIVALS:
        for lvl in LEVELS:
            r = run_negotiation(cfg, RECIPROCATOR, rival, info_level=lvl, rounds=rounds)
            rows.append({
                "rival": rival, "info_level": lvl,
                "reciprocator_profit": round(r.profit_solar, 1),
                "rival_profit": round(r.profit_wind, 1),
                "consumer_cost": round(r.buyer_cost, 1),
                "shortfall": round(r.shortfall, 3),
            })
            traj[(rival, lvl)] = [x["price_solar"] for x in r.rounds]
    return pd.DataFrame(rows), traj


def plot(cfg, df, traj, rounds: int = 12):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 4.3))

    # (1) Price trajectory of the reciprocator vs deception: full retaliates, blind is exploited.
    for lvl in ("full", "blind"):
        ax0.plot(range(rounds), traj[("deception", lvl)], marker="o",
                 color=LEVEL_COLORS[lvl], label=f"{lvl} feedback")
    ax0.axhline(cfg.price_max, ls=":", color="grey", lw=0.8)
    ax0.axhline(cfg.price_min, ls=":", color="grey", lw=0.8)
    ax0.set_xlabel("round"); ax0.set_ylabel("reciprocator asked price")
    ax0.set_title("Tit-for-tat vs deception:\nit only retaliates when it can see the undercut",
                  fontsize=9)
    ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)

    # (2) Reciprocator profit and (3) consumer cost by information level, per rival.
    x = np.arange(len(RIVALS))
    w = 0.26
    for k, lvl in enumerate(LEVELS):
        sub = df[df["info_level"] == lvl].set_index("rival")
        ax1.bar(x + (k - 1) * w, [sub.loc[r, "reciprocator_profit"] for r in RIVALS],
                width=w, color=LEVEL_COLORS[lvl], label=lvl)
        ax2.bar(x + (k - 1) * w, [sub.loc[r, "consumer_cost"] for r in RIVALS],
                width=w, color=LEVEL_COLORS[lvl], label=lvl)
    for ax, title, ylab in ((ax1, "Reciprocator profit", "solar (tit-for-tat) profit"),
                            (ax2, "Consumer cost", "buyer cost")):
        ax.set_xticks(x); ax.set_xticklabels(RIVALS, rotation=10)
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=9)
        ax.legend(title="feedback", fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Value of information — richer feedback lets the reciprocator enforce "
                 f"cooperation (scenario {cfg.label})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "mas_information.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamp", default="2017-06-15 09:00:00",
                    help="a symmetric scenario, so undercutting is partial (the case a "
                         "quantity-only signal cannot detect)")
    ap.add_argument("--rounds", type=int, default=12)
    args = ap.parse_args()

    cfg = scenario_from_timestamp(args.timestamp)
    print(f"Scenario {cfg.label}: demand={cfg.demand:.2f} solar={cfg.gen_solar:.2f} "
          f"wind={cfg.gen_wind:.2f} kW\n")

    df, traj = run_grid(cfg, rounds=args.rounds)
    print(df.to_string(index=False))

    # Headline: value of the rival-price signal to the reciprocator and the consumer.
    for rival in RIVALS:
        full = df[(df.rival == rival) & (df.info_level == "full")].iloc[0]
        blind = df[(df.rival == rival) & (df.info_level == "blind")].iloc[0]
        print(f"\nvs {rival}: rival-price signal is worth "
              f"{full.reciprocator_profit - blind.reciprocator_profit:+.1f} profit to the "
              f"reciprocator; consumer cost {blind.consumer_cost:.0f} (blind) -> "
              f"{full.consumer_cost:.0f} (full).")

    out = plot(cfg, df, traj, rounds=args.rounds)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "mas_information.csv", index=False)
    print(f"\nSaved -> {out}")
    print(f"Saved -> {RESULTS / 'mas_information.csv'}")


if __name__ == "__main__":
    main()
