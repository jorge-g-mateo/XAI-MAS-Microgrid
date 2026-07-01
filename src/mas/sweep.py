"""Multi-scenario sweep of the negotiation strategies.

A single timestep tells you what happens at one operating point; robust analysis needs
to show that the strategy conclusions (information hiding / deception hurt the
consumer, opponent modeling extracts margin while staying near the optimum) hold
*across conditions*, not just at one lucky hour.

This module runs the full (solar_strategy x wind_strategy) experiment over a grid
of contrasting scenarios — different hours of the day (solar peak vs. night) and
different seasons — then aggregates per strategy pair. Outputs:

  * ``results/mas_sweep_raw.csv``        every (scenario x strategy-pair) row
  * ``results/mas_sweep_aggregate.csv``  per strategy-pair means across scenarios
  * ``results/mas_sweep_buyer_cost.png`` 4x4 heatmap of mean buyer cost
  * ``results/mas_sweep_shortfall.png``  4x4 heatmap of mean shortfall

Usage:
    python -m src.mas.sweep
    python -m src.mas.sweep --year 2018 --rounds 10 --no-optimum
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for Docker/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.inference import add_date_features, load_solar_data
from src.mas.run_mas import run_experiment
from src.mas.strategies import STRATEGIES
from src.optimization.market import scenario_from_timestamp

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# Contrasting conditions: night/morning/solar-peak/evening x four seasons.
DEFAULT_HOURS = [3, 9, 13, 19]
DEFAULT_MONTHS = [1, 4, 7, 10]


def select_scenarios(year: int = 2018,
                     hours: list[int] | None = None,
                     months: list[int] | None = None) -> list[str]:
    """Pick representative ``Date`` strings present in the datasets.

    For every (month, hour) pair we take the first matching timestamp in ``year``,
    giving a small grid that spans day/night and seasonal generation regimes.
    """
    hours = hours or DEFAULT_HOURS
    months = months or DEFAULT_MONTHS

    df = add_date_features(load_solar_data())
    df["_dt"] = pd.to_datetime(df["Date"])
    df = df[df["_dt"].dt.year == year]

    scenarios: list[str] = []
    for month in months:
        for hour in hours:
            match = df[(df["_dt"].dt.month == month) & (df["hour"] == hour)]
            if not match.empty:
                scenarios.append(str(match.iloc[0]["Date"]))
    return scenarios


def run_sweep(year: int = 2018, rounds: int = 10, with_optimum: bool = True,
              hours: list[int] | None = None,
              months: list[int] | None = None) -> pd.DataFrame:
    """Run the strategy experiment over every selected scenario; return raw rows."""
    scenarios = select_scenarios(year, hours, months)
    frames = []
    for i, ts in enumerate(scenarios, 1):
        cfg = scenario_from_timestamp(ts)
        print(f"[{i}/{len(scenarios)}] {ts}: demand={cfg.demand:.2f} "
              f"solar={cfg.gen_solar:.2f} wind={cfg.gen_wind:.2f} kW")
        df = run_experiment(cfg, rounds=rounds, with_optimum=with_optimum)
        df.insert(0, "scenario", ts)
        df["demand_kw"] = cfg.demand
        df["gen_solar_kw"] = cfg.gen_solar
        df["gen_wind_kw"] = cfg.gen_wind
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """Average each strategy pair's outcome across all scenarios."""
    agg = {
        "buyer_cost": "mean",
        "profit_solar": "mean",
        "profit_wind": "mean",
        "shortfall": "mean",
    }
    if "dist_to_pareto" in raw.columns:
        agg["dist_to_pareto"] = "mean"
    out = raw.groupby(["solar_strategy", "wind_strategy"], as_index=False).agg(agg)
    if "dominated_by_optimum" in raw.columns:
        dom = (raw.groupby(["solar_strategy", "wind_strategy"])["dominated_by_optimum"]
               .mean().reset_index(name="frac_dominated"))
        out = out.merge(dom, on=["solar_strategy", "wind_strategy"])
    return out.round(3)


def _heatmap(agg: pd.DataFrame, value: str, title: str, fname: str,
             cmap: str, lower_is_better: bool):
    names = list(STRATEGIES)
    grid = (agg.pivot(index="solar_strategy", columns="wind_strategy", values=value)
            .reindex(index=names, columns=names))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(grid.values, cmap=cmap)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("wind strategy (AE)"); ax.set_ylabel("solar strategy (AS)")
    vmax = np.nanmax(grid.values)
    for i in range(len(names)):
        for j in range(len(names)):
            v = grid.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}",
                        ha="center", va="center",
                        color="white" if v > 0.6 * vmax else "black", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=f"{value} ({'lower' if lower_is_better else 'higher'} is better)")
    fig.tight_layout()
    fig.savefig(RESULTS / fname, dpi=130)
    plt.close(fig)


def save_outputs(raw: pd.DataFrame, agg: pd.DataFrame):
    RESULTS.mkdir(exist_ok=True)
    raw.round(3).to_csv(RESULTS / "mas_sweep_raw.csv", index=False)
    agg.to_csv(RESULTS / "mas_sweep_aggregate.csv", index=False)
    _heatmap(agg, "buyer_cost",
             "Mean buyer cost across scenarios (AC pays)",
             "mas_sweep_buyer_cost.png", "coolwarm", lower_is_better=True)
    _heatmap(agg, "shortfall",
             "Mean unmet demand across scenarios (cooperation failure)",
             "mas_sweep_shortfall.png", "Reds", lower_is_better=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--no-optimum", action="store_true",
                    help="skip the per-scenario Pareto-front comparison (faster)")
    args = ap.parse_args()

    raw = run_sweep(year=args.year, rounds=args.rounds,
                    with_optimum=not args.no_optimum)
    agg = aggregate(raw)
    save_outputs(raw, agg)

    pd.set_option("display.width", 200)
    print(f"\nSwept {raw['scenario'].nunique()} scenarios x "
          f"{len(STRATEGIES)**2} strategy pairs = {len(raw)} runs.\n")
    print("Per strategy-pair averages (sorted by buyer cost):")
    print(agg.sort_values("buyer_cost").to_string(index=False))

    base = agg[(agg.solar_strategy == "honest") & (agg.wind_strategy == "honest")].iloc[0]
    worst = agg.loc[agg.shortfall.idxmax()]
    print(f"\nBaseline honest/honest: mean buyer_cost={base.buyer_cost:.2f}, "
          f"mean shortfall={base.shortfall:.3f}")
    print(f"Worst mean shortfall: {worst.shortfall:.3f} kW with "
          f"({worst.solar_strategy}/{worst.wind_strategy})")
    print(f"\nOutputs in {RESULTS}/ (mas_sweep_*.{{csv,png}})")


if __name__ == "__main__":
    main()
