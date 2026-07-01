"""Game-theoretic analysis of the seller strategy game (M3) ⭐.

The (solar_strategy × wind_strategy) experiment is a **bimatrix game**: the two
generators are the players, their strategy sets are the negotiation strategies
(``STRATEGIES`` — the four static strategies plus the repeated-game
``tit_for_tat``, a 5×5 grid), and the payoffs are their mean per-round profits.
This module does the formal analysis the SMA exam asks for:

  * build the payoff bimatrix (π_solar, π_wind);
  * compute each player's **best response** to the other;
  * find the **pure-strategy Nash equilibria** (cells where neither player can
    improve unilaterally);
  * find the **social optimum** (the cell that minimizes the consumer's cost);
  * contrast them — when the Nash equilibrium is *not* the social optimum the
    market is a **partial-conflict game** (an empirical prisoner's dilemma):
    individually rational behaviour is collectively worse for the system.

Outputs (results/):
  - mas_payoff_matrix.csv   the bimatrix + best-response / NE / social-optimum flags
  - mas_nash.png            the payoff grid with the NE and the social optimum marked

Usage:
    python -m src.mas.game_analysis --timestamp "2017-06-15 19:00:00"
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
from src.mas.strategies import BASELINE_STRATEGIES, STRATEGIES
from src.optimization.market import scenario_from_timestamp

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
TOL = 1e-6  # relative tolerance for "is a best response" ties


def build_payoffs(cfg, rounds: int = 12, buyer_strategy: str = "price_taker"):
    """Run every (solar × wind) strategy pair → payoff matrices and cost matrix.

    The payoff of a strategy pair is the **mean per-round** profit (and cost),
    the correct measure for this *repeated* game: it averages out the transient
    of adaptive strategies and the oscillation of the reciprocal (tit-for-tat)
    strategy, so the matrix entry is a steady-state value rather than a single
    final-round snapshot. For the converging static strategies the mean equals
    the final round (the NE is unchanged), but it is what makes a history-
    dependent strategy's payoff well-defined as a single matrix entry.

    The bimatrix is the game **between the two sellers**, so the consumer is held
    fixed (``buyer_strategy``) to isolate the sellers' strategic interaction. The
    default is the passive ``price_taker``, the cleanest baseline where the
    prisoner's dilemma appears; re-running with a bargaining consumer shows how a
    negotiating buyer reshapes (and can mitigate) the sellers' equilibrium.
    """
    names = list(STRATEGIES)
    n = len(names)
    pi_s = np.zeros((n, n))   # solar profit (row player)
    pi_w = np.zeros((n, n))   # wind profit (column player)
    cost = np.zeros((n, n))
    short = np.zeros((n, n))
    for i, s in enumerate(names):
        for j, w in enumerate(names):
            r = run_negotiation(cfg, s, w, buyer_strategy=buyer_strategy, rounds=rounds)
            pi_s[i, j] = float(np.mean([x["profit_solar"] for x in r.rounds]))
            pi_w[i, j] = float(np.mean([x["profit_wind"] for x in r.rounds]))
            cost[i, j] = float(np.mean([x["cost"] for x in r.rounds]))
            short[i, j] = float(np.mean([x["shortfall"] for x in r.rounds]))
    return names, pi_s, pi_w, cost, short


def analyze(names, pi_s, pi_w, cost):
    """Best responses, pure-strategy Nash equilibria and the social optimum."""
    n = len(names)
    # Solar (row player) best response to each wind column j: argmax over rows.
    solar_br = [pi_s[:, j] >= pi_s[:, j].max() - TOL * (abs(pi_s[:, j].max()) + 1)
                for j in range(n)]  # solar_br[j] is a boolean mask over rows
    # Wind (column player) best response to each solar row i: argmax over columns.
    wind_br = [pi_w[i, :] >= pi_w[i, :].max() - TOL * (abs(pi_w[i, :].max()) + 1)
               for i in range(n)]

    nash = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if solar_br[j][i] and wind_br[i][j]:
                nash[i, j] = True

    # Social optimum = minimum consumer cost (ties → also lowest = best for AC).
    so_idx = np.unravel_index(np.argmin(cost), cost.shape)
    return nash, so_idx, solar_br, wind_br


def save_outputs(names, pi_s, pi_w, cost, short, nash, so_idx, timestamp,
                 buyer_strategy: str = "price_taker"):
    RESULTS.mkdir(exist_ok=True)
    n = len(names)
    # price_taker keeps the canonical filenames; other consumers get a suffix so
    # the Nash-shift-across-consumers story has one figure/CSV each.
    suffix = "" if buyer_strategy == "price_taker" else f"_{buyer_strategy}"

    rows = []
    for i, s in enumerate(names):
        for j, w in enumerate(names):
            rows.append({
                "buyer_strategy": buyer_strategy,
                "solar_strategy": s, "wind_strategy": w,
                "profit_solar": round(pi_s[i, j], 2), "profit_wind": round(pi_w[i, j], 2),
                "buyer_cost": round(cost[i, j], 2), "shortfall": round(short[i, j], 3),
                "nash_equilibrium": bool(nash[i, j]),
                "social_optimum": (i, j) == tuple(so_idx),
            })
    pd.DataFrame(rows).to_csv(RESULTS / f"mas_payoff_matrix{suffix}.csv", index=False)

    # Grid: color by consumer cost, annotate "πs / πw", mark NE and social optimum.
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cost, cmap="RdYlGn_r", aspect="auto")
    fig.colorbar(im, ax=ax, label="consumer cost (lower = better for AC)")
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    ax.set_xlabel("Wind seller strategy (player 2)")
    ax.set_ylabel("Solar seller strategy (player 1)")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{pi_s[i, j]:.0f} / {pi_w[i, j]:.0f}",
                    ha="center", va="center", fontsize=8, color="black")
            if nash[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="red", lw=3))
            if (i, j) == tuple(so_idx):
                ax.add_patch(plt.Rectangle((j - 0.42, i - 0.42), 0.84, 0.84, fill=False,
                                           edgecolor="blue", lw=2, ls="--"))
    ax.set_title("Seller payoff bimatrix (π_solar / π_wind) — "
                 f"consumer: {buyer_strategy}\n"
                 "red = Nash equilibrium · blue dashed = social optimum (min AC cost)")
    fig.tight_layout(); fig.savefig(RESULTS / f"mas_nash{suffix}.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamp", default="2017-06-15 19:00:00")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--buyer", default="price_taker",
                    help="consumer strategy held fixed for the seller bimatrix game")
    args = ap.parse_args()

    cfg = scenario_from_timestamp(args.timestamp)
    print(f"Scenario {cfg.label}: demand={cfg.demand:.2f} solar={cfg.gen_solar:.2f} "
          f"wind={cfg.gen_wind:.2f} kW  (consumer={args.buyer})\n")

    names, pi_s, pi_w, cost, short = build_payoffs(cfg, rounds=args.rounds,
                                                   buyer_strategy=args.buyer)
    nash, so_idx, _, _ = analyze(names, pi_s, pi_w, cost)
    save_outputs(names, pi_s, pi_w, cost, short, nash, so_idx, args.timestamp,
                 buyer_strategy=args.buyer)

    ne_cells = [(names[i], names[j]) for i in range(len(names)) for j in range(len(names)) if nash[i, j]]
    so = (names[so_idx[0]], names[so_idx[1]])
    print("Pure-strategy Nash equilibria:", ne_cells or "none (mixed only)")
    print(f"Social optimum (min consumer cost): {so}  cost={cost[so_idx]:.2f}")

    # Baseline sub-game: restrict to the four original M3 strategies (Part I), so
    # the documented NE is recoverable however many Part-II strategies are added.
    base = [k for k, nm in enumerate(names) if nm in BASELINE_STRATEGIES]
    if 0 < len(base) < len(names):
        bn = [names[k] for k in base]
        sub = np.ix_(base, base)
        b_nash, b_so, _, _ = analyze(bn, pi_s[sub], pi_w[sub], cost[sub])
        b_ne = [(bn[i], bn[j]) for i in range(len(bn)) for j in range(len(bn)) if b_nash[i, j]]
        print(f"  [baseline 4-strategy sub-game] Nash: {b_ne or 'none'}  "
              f"social optimum: {(bn[b_so[0]], bn[b_so[1]])}")
    nash_is_social = all(c == so for c in ne_cells) and len(ne_cells) == 1
    if ne_cells and not nash_is_social:
        print("-> Partial-conflict game: the Nash equilibrium is NOT the social optimum "
              "(empirical prisoner's dilemma -- individually rational, collectively worse).")
    elif nash_is_social:
        print("-> The Nash equilibrium coincides with the social optimum (aligned incentives).")
    print(f"\nOutputs -> {RESULTS}/ (mas_payoff_matrix.csv, mas_nash.png)")


if __name__ == "__main__":
    main()
