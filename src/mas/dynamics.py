"""Negotiation dynamics over rounds.

The Contract Net runs for several rounds so adaptive strategies can react to the
feedback. This script visualizes that temporal behaviour for a chosen strategy
pairing: how the sellers' asked prices and per-round profits evolve. It makes the
repeated-game nature of the negotiation explicit (e.g. mutual opponent modeling
climbing toward the price ceiling, an instance of history-exploiting play).

Outputs (written to ``results/``):
  - ``mas_dynamics.png``   two panels (prices, per-round profit) over rounds

Usage:
    python -m src.mas.dynamics --solar opponent_modeling --wind opponent_modeling
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.mas.simulation import run_negotiation
from src.optimization.market import scenario_from_timestamp

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def plot_dynamics(cfg, solar_strategy: str, wind_strategy: str, rounds: int = 12):
    res = run_negotiation(cfg, solar_strategy, wind_strategy, rounds=rounds)
    rs = res.rounds
    r = [x["round"] for x in rs]
    ps = [x["price_solar"] for x in rs]
    pw = [x["price_wind"] for x in rs]
    prof_s = [x["profit_solar"] for x in rs]
    prof_w = [x["profit_wind"] for x in rs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(r, ps, marker="o", label="price AS (solar)", color="tab:orange")
    ax1.plot(r, pw, marker="s", label="price AE (wind)", color="tab:blue")
    ax1.axhline(cfg.price_max, ls="--", color="grey", lw=0.8, label="price band")
    ax1.axhline(cfg.price_min, ls="--", color="grey", lw=0.8)
    ax1.set_xlabel("round"); ax1.set_ylabel("asked unit price")
    ax1.set_title("Asked prices over rounds")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2.plot(r, prof_s, marker="o", label="profit AS", color="tab:orange")
    ax2.plot(r, prof_w, marker="s", label="profit AE", color="tab:blue")
    ax2.set_xlabel("round"); ax2.set_ylabel("per-round profit")
    ax2.set_title("Per-round profit over rounds")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Negotiation dynamics — {solar_strategy} (AS) vs. {wind_strategy} (AE), "
                 f"scenario {cfg.label}")
    fig.tight_layout()
    out = RESULTS / "mas_dynamics.png"
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return res, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamp", type=str, default="2017-06-15 19:00:00")
    ap.add_argument("--solar", type=str, default="opponent_modeling")
    ap.add_argument("--wind", type=str, default="opponent_modeling")
    ap.add_argument("--rounds", type=int, default=12)
    args = ap.parse_args()

    cfg = scenario_from_timestamp(args.timestamp)
    res, out = plot_dynamics(cfg, args.solar, args.wind, rounds=args.rounds)
    print(f"Scenario {cfg.label}: demand={cfg.demand:.2f} "
          f"solar={cfg.gen_solar:.2f} wind={cfg.gen_wind:.2f} kW")
    print(f"Final prices: AS={res.final_price_solar}, AE={res.final_price_wind}")
    print(f"Final profits: AS={res.profit_solar:.2f}, AE={res.profit_wind:.2f}, "
          f"buyer_cost={res.buyer_cost:.2f}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
