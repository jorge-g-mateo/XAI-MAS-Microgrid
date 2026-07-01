"""Opponent modeling: ad-hoc tâtonnement vs. Bayesian inference (M4).

The brief lists **opponent modeling** as a strategy to analyze and aspires to
"estimation by regression / Bayesian updating over past offers". This script
contrasts the two opponent-modelers we ship:

  * ``opponent_modeling``  — the ad-hoc *tâtonnement*: react to the last
    clearing (push up when winning, undercut when losing);
  * ``bayesian_opponent``  — a Normal–Normal posterior over the rival's price,
    updated from the whole history of observed offers, pricing a
    confidence-scaled undercut (M4).

It produces two artifacts:

  1. **Belief convergence** (``mas_opponent_belief.png``): the Bayesian seller's
     posterior mean ± a 2σ confidence band over rounds, against the rival's
     actually-observed price — for a *fixed* rival (info_hiding) and an
     *adaptive* one (opponent_modeling, a moving target). Shows *how* the
     opponent is modeled, not merely *that* it is.
  2. **Head-to-head** (``mas_opponent_comparison.{csv,png}``): simple vs Bayesian
     opponent-modeler as the solar seller against a panel of rivals — cumulative
     profit and final price-estimation error.

Usage:
    python -m src.mas.opponent_modeling_experiment [--timestamp "2017-06-15 09:00:00"]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.mas.acl import MessageBus
from src.mas.agents import ConsumerAgent, SellerAgent
from src.mas.buyer_strategies import make_buyer_strategy
from src.mas.strategies import BayesianOpponentModelingStrategy, Strategy, make_strategy
from src.optimization.market import scenario_from_timestamp

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

SOLAR, WIND, CONSUMER = "AS_solar", "AE_wind", "AC_consumer"


def _run(cfg, solar_strategy: Strategy, wind_name: str, rounds: int = 12):
    """Run a negotiation with a given solar *strategy instance*; return (agent, rounds)."""
    bus = MessageBus()
    ss = SellerAgent(SOLAR, cfg.gen_solar, cfg.mc_solar, solar_strategy,
                     cfg.price_min, cfg.price_max)
    sw = SellerAgent(WIND, cfg.gen_wind, cfg.mc_wind, make_strategy(wind_name),
                     cfg.price_min, cfg.price_max)
    cons = ConsumerAgent(CONSUMER, cfg.demand, bus, make_buyer_strategy("price_taker"),
                         cfg.price_min, cfg.price_max, sellers=[SOLAR, WIND])
    bus.register(SOLAR, ss.receive)
    bus.register(WIND, sw.receive)
    recs = [cons.run_round(r) for r in range(rounds)]
    return ss, recs


def plot_belief(cfg, rounds: int = 12):
    """Posterior mean ± 2σ vs. the observed rival price, for fixed & adaptive rivals."""
    rivals = [("info_hiding", "fixed-price rival (info_hiding)"),
              ("opponent_modeling", "adaptive rival (opponent_modeling)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, (rival, title) in zip(axes, rivals):
        bayes = BayesianOpponentModelingStrategy()
        agent, _ = _run(cfg, bayes, rival, rounds=rounds)
        rs = [r for r, _, _ in bayes.trace]
        mu = [m for _, m, _ in bayes.trace]
        sd = [s for _, _, s in bayes.trace]
        lo = [m - 2 * s for m, s in zip(mu, sd)]
        hi = [m + 2 * s for m, s in zip(mu, sd)]
        observed = [fb.rival_price for fb in bayes.history]  # rival price seen each round

        ax.fill_between(rs, lo, hi, color="tab:purple", alpha=0.18, label="posterior ±2σ")
        ax.plot(rs, mu, color="tab:purple", marker="o", label="posterior mean μ (belief)")
        ax.plot(range(len(observed)), observed, color="tab:green", ls="--", marker="x",
                label="observed rival price")
        ax.axhline(cfg.price_max, ls=":", color="grey", lw=0.8)
        ax.axhline(cfg.price_min, ls=":", color="grey", lw=0.8)
        ax.set_xlabel("round")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("unit price")
    fig.suptitle("Bayesian opponent modeling — belief converges to the rival's price "
                 f"(scenario {cfg.label})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "mas_opponent_belief.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def head_to_head(cfg, rounds: int = 12):
    """Simple vs Bayesian opponent-modeler (as solar) against a panel of rivals."""
    rivals = ["honest", "info_hiding", "deception", "opponent_modeling", "tit_for_tat"]
    rows = []
    for rival in rivals:
        # Simple tâtonnement modeler.
        simple, _ = _run(cfg, make_strategy("opponent_modeling"), rival, rounds=rounds)
        # Bayesian modeler (keep the instance for its final belief).
        bayes_strat = BayesianOpponentModelingStrategy()
        bayes, _ = _run(cfg, bayes_strat, rival, rounds=rounds)
        # Compare the belief to the MEAN observed rival price (fair for a stationary
        # rival, and for an oscillating one like tit_for_tat the posterior estimates
        # the mean rather than the last round's value).
        obs = [fb.rival_price for fb in bayes_strat.history
               if fb.demand - fb.sold > 1e-9]  # rounds where the rival actually sold
        obs_mean = sum(obs) / len(obs) if obs else float("nan")
        est_err = abs(bayes_strat.mu - obs_mean)
        rows.append({
            "rival": rival,
            "profit_simple": round(simple.profit, 1),
            "profit_bayesian": round(bayes.profit, 1),
            "bayes_belief_mu": round(bayes_strat.mu, 2),
            "rival_price_mean": round(obs_mean, 2),
            "belief_error": round(est_err, 3),
        })
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "mas_opponent_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = range(len(rivals))
    w = 0.38
    ax.bar([i - w / 2 for i in x], df["profit_simple"], width=w,
           label="simple (tâtonnement)", color="tab:gray")
    ax.bar([i + w / 2 for i in x], df["profit_bayesian"], width=w,
           label="Bayesian (M4)", color="tab:purple")
    ax.set_xticks(list(x))
    ax.set_xticklabels(rivals, rotation=20, ha="right")
    ax.set_ylabel("solar seller cumulative profit")
    ax.set_title("Opponent modeling head-to-head: ad-hoc vs Bayesian "
                 f"(as solar seller, scenario {cfg.label})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = RESULTS / "mas_opponent_comparison.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return df, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamp", default="2017-06-15 09:00:00")
    ap.add_argument("--rounds", type=int, default=12)
    args = ap.parse_args()

    cfg = scenario_from_timestamp(args.timestamp)
    print(f"Scenario {cfg.label}: demand={cfg.demand:.2f} solar={cfg.gen_solar:.2f} "
          f"wind={cfg.gen_wind:.2f} kW\n")

    belief_out = plot_belief(cfg, rounds=args.rounds)
    df, comp_out = head_to_head(cfg, rounds=args.rounds)
    print(df.to_string(index=False))
    print(f"\nSaved -> {belief_out}")
    print(f"Saved -> {comp_out}")
    print(f"Saved -> {RESULTS / 'mas_opponent_comparison.csv'}")


if __name__ == "__main__":
    main()
