"""Temporal battery agent — storage arbitrage over one day (M1, MAS side).

The static market cannot give a battery any profit: buying only costs money and
there is no "later" to sell into. The battery's value is **temporal** — buy when
energy is cheap and abundant (midday solar surplus), sell when it is scarce and
expensive (evening/night) — so this runs a real **24-hour sequence** with a state
of charge that persists between hours.

Why a fixed demand profile (not demand = 0.8·gen): with demand tied to generation
the relative scarcity is constant every hour and the price never moves, so there
is nothing to arbitrage. We therefore use an (independent) constant load, which
makes midday a surplus (cheap) and night a deficit (expensive) — the price spread
the battery lives on.

Price signal: scarcity-based, ``price = p_min + (p_max−p_min)·clip((demand−gen)/demand, 0, 1)``
→ p_min when there is surplus, rising to p_max as generation vanishes. The battery
follows a **threshold rule** (the "bank"): charge from the surplus when the price
is low; discharge into the deficit when the price clears its cost basis by a
margin. The consumer is served first (it values energy more), so the battery
charges the *surplus* — the competition for cheap energy resolves in the
consumer's favour, leaving the battery the leftover.

Outputs (results/):
  - mas_battery_day.csv    per-hour gen/demand/price/SoC/charge/discharge/cashflow
  - mas_battery_day.png    SoC + price + flows over the day

Usage:
    python -m src.mas.battery --date 2017-06-15
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.optimization.market import scenario_from_timestamp

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# Evening-peaked daily load profile (normalized to mean ≈ 1): low at night, a
# small morning shoulder, a dip at midday (when solar is plentiful) and the peak
# in the evening (18-21h, when solar is gone). This is what makes storage pay:
# midday surplus is cheap, the evening peak is scarce and expensive.
DEFAULT_LOAD_PROFILE = np.array([
    0.55, 0.50, 0.48, 0.48, 0.50, 0.60,   # 00-05 night
    0.75, 0.90, 0.85, 0.70, 0.62, 0.60,   # 06-11 morning → midday dip (solar covers)
    0.60, 0.62, 0.70, 0.95, 1.35, 1.75,   # 12-17 afternoon ramp
    2.00, 1.95, 1.65, 1.20, 0.85, 0.65,   # 18-23 evening peak → taper
])
DEFAULT_LOAD_PROFILE = DEFAULT_LOAD_PROFILE / DEFAULT_LOAD_PROFILE.mean()  # mean = 1


@dataclass
class BatteryAgent:
    """A storage prosumer that arbitrages the price across the day."""

    capacity: float                 # kWh usable
    charge_rate: float              # max kW charged per hour
    discharge_rate: float           # max kW discharged per hour
    efficiency: float = 0.92        # round-trip (applied on charge)
    buy_ratio: float = 0.25         # buy when price <= p_min + buy_ratio*band
    margin: float = 0.05            # require sell price >= avg cost * (1 + margin)
    soc: float = 0.0                # state of charge (kWh)
    # bookkeeping
    cost_basis: float = 0.0         # cost of the energy CURRENTLY stored (FIFO-ish avg)
    revenue: float = 0.0            # money earned discharging
    spent: float = 0.0              # money spent charging (incl. still-stored inventory)
    cost_sold: float = 0.0          # cost basis of the energy actually sold
    shortfall_avoided: float = 0.0

    @property
    def avg_cost(self) -> float:
        return self.cost_basis / self.soc if self.soc > 1e-9 else 0.0

    @property
    def profit(self) -> float:
        """Realized arbitrage profit = revenue − cost of the energy actually sold.
        Energy still stored at the end is inventory (valued at ``cost_basis``), not
        a loss."""
        return self.revenue - self.cost_sold

    @property
    def inventory_value(self) -> float:
        return self.cost_basis

    def step(self, price: float, surplus: float, deficit: float,
             p_min: float, p_max: float) -> dict:
        """One hour: maybe charge from surplus (cheap) or discharge into deficit
        (expensive). Returns the hour's action for logging."""
        buy_threshold = p_min + self.buy_ratio * (p_max - p_min)
        sell_threshold = max(self.avg_cost * (1 + self.margin),
                             p_min + self.buy_ratio * (p_max - p_min))
        charge = discharge = 0.0

        if price <= buy_threshold and self.soc < self.capacity and surplus > 0:
            charge = min(self.charge_rate, self.capacity - self.soc, surplus)
            stored = charge * self.efficiency
            self.soc += stored
            self.cost_basis += charge * price   # paid for what we drew from the grid
            self.spent += charge * price
        elif price >= sell_threshold and self.soc > 0 and deficit > 0:
            discharge = min(self.discharge_rate, self.soc, deficit)
            # release the proportional cost basis with the energy sold
            released = self.avg_cost * discharge
            self.cost_basis -= released
            self.cost_sold += released
            self.soc -= discharge
            self.revenue += discharge * price
            self.shortfall_avoided += discharge

        return {"charge": charge, "discharge": discharge, "soc": self.soc,
                "buy_threshold": round(buy_threshold, 2),
                "sell_threshold": round(sell_threshold, 2)}


def _scarcity_price(gen_total: float, demand: float, p_min: float, p_max: float) -> float:
    scarcity = np.clip((demand - gen_total) / demand, 0.0, 1.0) if demand > 0 else 0.0
    return p_min + (p_max - p_min) * scarcity


def run_battery_day(date: str = "2017-06-15", mean_demand: float | None = None,
                    load_profile: np.ndarray | None = None,
                    p_min: float = 50.0, p_max: float = 80.0,
                    battery: BatteryAgent | None = None) -> pd.DataFrame:
    """Simulate the 24 hours of ``date`` with a battery arbitraging the price.

    Per-hour demand = ``mean_demand`` × an evening-peaked load profile, so the day
    is roughly balanced overall but mismatched in time (midday surplus / evening
    deficit) — the spread the battery monetizes. ``mean_demand`` defaults to the
    day's mean total generation."""
    # Generation per hour from the predictive models (kW).
    gens = []
    for h in range(24):
        ts = f"{date} {h:02d}:00:00"
        cfg = scenario_from_timestamp(ts)
        gens.append((ts, cfg.gen_solar, cfg.gen_wind))
    gen_total = np.array([gs + gw for _, gs, gw in gens])

    if mean_demand is None:
        mean_demand = float(gen_total.mean())  # balanced over the day
    if load_profile is None:
        load_profile = DEFAULT_LOAD_PROFILE
    demand_h = mean_demand * load_profile      # per-hour demand (evening-peaked)

    if battery is None:
        cap = max(1.0, float(gen_total.max()) * 3)  # a few hours of peak generation
        battery = BatteryAgent(capacity=cap, charge_rate=cap / 4, discharge_rate=cap / 4)

    rows = []
    for ((ts, gs, gw), gt, d) in zip(gens, gen_total, demand_h):
        price = _scarcity_price(gt, d, p_min, p_max)
        surplus = max(0.0, gt - d)
        deficit = max(0.0, d - gt)
        act = battery.step(price, surplus, deficit, p_min, p_max)
        # Net deficit the consumer still faces after the battery discharges.
        net_deficit = max(0.0, deficit - act["discharge"])
        rows.append({
            "time": ts[11:16], "gen_total": round(gt, 3), "demand": round(d, 3),
            "price": round(price, 2), "surplus": round(surplus, 3),
            "deficit": round(deficit, 3), "charge": round(act["charge"], 3),
            "discharge": round(act["discharge"], 3), "soc": round(act["soc"], 3),
            "net_deficit": round(net_deficit, 3),
        })
    df = pd.DataFrame(rows)
    df.attrs["battery"] = battery
    df.attrs["mean_demand"] = mean_demand
    return df


def save_outputs(df: pd.DataFrame, date: str):
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "mas_battery_day.csv", index=False)
    bat: BatteryAgent = df.attrs["battery"]

    x = range(len(df))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Panel A: generation, demand, price.
    ax1.plot(x, df["gen_total"], color="tab:green", marker=".", label="generation (kW)")
    ax1.plot(x, df["demand"], color="tab:red", ls="--", lw=1.2, label="demand (kW)")
    ax1.set_ylabel("power (kW)")
    axp = ax1.twinx()
    axp.plot(x, df["price"], color="tab:purple", lw=1.5, alpha=0.7, label="price")
    axp.set_ylabel("price", color="tab:purple")
    ax1.set_title(f"Battery arbitrage over {date} — profit={bat.profit:.1f}, "
                  f"shortfall avoided={bat.shortfall_avoided:.2f} kWh")
    ax1.legend(loc="upper left", fontsize=8)

    # Panel B: SoC + charge/discharge flows.
    ax2.bar(x, df["charge"], color="tab:blue", alpha=0.6, label="charge")
    ax2.bar(x, -df["discharge"], color="tab:orange", alpha=0.8, label="discharge")
    axs = ax2.twinx()
    axs.plot(x, df["soc"], color="black", marker=".", lw=1.5, label="state of charge")
    axs.set_ylabel("SoC (kWh)")
    ax2.set_ylabel("charge / discharge (kW)")
    ax2.axhline(0, color="grey", lw=0.6)
    ax2.set_xticks(list(x)); ax2.set_xticklabels(df["time"], rotation=90, fontsize=7)
    ax2.set_xlabel("hour")
    ax2.legend(loc="upper left", fontsize=8); axs.legend(loc="upper right", fontsize=8)

    fig.suptitle("Temporal storage agent: charge the cheap surplus, discharge into the costly deficit")
    fig.tight_layout()
    fig.savefig(RESULTS / "mas_battery_day.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2017-06-15")
    args = ap.parse_args()

    df = run_battery_day(args.date)
    save_outputs(df, args.date)
    bat: BatteryAgent = df.attrs["battery"]
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"\nBattery: capacity={bat.capacity:.2f} kWh  realized_profit={bat.profit:.2f}  "
          f"revenue={bat.revenue:.2f}  cost_of_sold={bat.cost_sold:.2f}  "
          f"(unsold inventory: {bat.soc:.2f} kWh worth {bat.inventory_value:.2f})  "
          f"shortfall_avoided={bat.shortfall_avoided:.2f} kWh")
    print(f"Outputs -> {RESULTS}/ (mas_battery_day.{{csv,png}})")


if __name__ == "__main__":
    main()
