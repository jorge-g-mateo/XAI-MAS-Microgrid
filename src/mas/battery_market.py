"""Per-agent storage in a 24-hour rolling negotiation (M1, MAS side — overhaul).

The single-instant market cannot give a battery any value: there is no "later" to
sell into. The earlier ``battery.py`` showed temporal arbitrage with one *external*
storage agent; this module instead gives **a battery to each of the three
negotiating agents** (the two sellers and the consumer) and rolls the FIPA-ACL
negotiation over a full day with a **state of charge that persists between hours**.
That temporal coupling is what makes the batteries change the negotiation itself:

  * a **seller** stores its zero-marginal-cost surplus when the hour is cheap
    (abundant generation) and discharges it into its offer when the hour is scarce
    and dear — time-shifting its sales up the price curve;
  * the **consumer** charges (buys a little extra) when energy is cheap and
    discharges to cover its own demand when energy is dear — shrinking the demand it
    must clear at the evening peak, which is exactly when sellers can extract surplus.

Each battery is operated within a **20–80 % state-of-charge band** (depth of
discharge ≤ 60 %), the standard operating window that avoids the accelerated
LiFePO4 aging seen at high/low SoC; round-trip efficiency 0.92 and a C/4 power
rating (energy:power ≈ 4 h) are typical Li-ion values. See ROADMAP III.1 for the
state-of-the-art references. Giving each prosumer its own storage + controller
mirrors P2P/local-energy-market designs where every prosumer schedules its battery.

Outputs (``results/``):
  - ``mas_battery_market_day.csv``   per-hour gen/demand/price/SoC/flows, both regimes
  - ``mas_battery_market.png``       SoC bands + daily cost/profit, with vs without storage

Usage:
    python -m src.mas.battery_market --date 2017-06-15
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.mas.battery import DEFAULT_LOAD_PROFILE, _scarcity_price
from src.mas.simulation import run_negotiation
from src.optimization.market import MarketConfig, scenario_from_timestamp

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


@dataclass
class AgentBattery:
    """A battery operated within a 20–80 % SoC band (a controller constraint).

    ``soc`` is absolute energy (kWh). Charge/discharge are clamped so the SoC never
    leaves ``[soc_min_frac, soc_max_frac] * capacity`` — the depth-of-discharge
    limit that protects cycle life. Round-trip loss is applied on charge."""

    capacity: float
    charge_rate: float
    discharge_rate: float
    efficiency: float = 0.92
    soc_min_frac: float = 0.20
    soc_max_frac: float = 0.80
    soc: float = 0.0

    def __post_init__(self):
        # Start at the bottom of the band (no usable charge yet → must buy/store first).
        self.soc = max(self.soc, self.soc_min_frac * self.capacity)

    @property
    def floor(self) -> float:
        return self.soc_min_frac * self.capacity

    @property
    def ceil(self) -> float:
        return self.soc_max_frac * self.capacity

    def chargeable(self, offered_energy: float) -> float:
        """How much grid energy it can absorb now (band headroom, rate, availability)."""
        headroom = max(0.0, self.ceil - self.soc) / self.efficiency
        return max(0.0, min(self.charge_rate, headroom, offered_energy))

    def dischargeable(self, wanted_energy: float) -> float:
        """How much it can deliver now (usable charge above the band floor, rate)."""
        usable = max(0.0, self.soc - self.floor)
        return max(0.0, min(self.discharge_rate, usable, wanted_energy))

    def charge(self, grid_energy: float) -> None:
        self.soc = min(self.ceil, self.soc + grid_energy * self.efficiency)

    def discharge(self, energy: float) -> None:
        self.soc = max(self.floor, self.soc - energy)


def _schedule_batteries(bat_s, bat_w, bat_c, gs, gw, load):
    """Physically schedule the three batteries for one hour by surplus/deficit.

    *Deficit hour* (load > generation, e.g. the evening peak): discharge to cover
    it — the consumer self-supplies first (cheapest for it), then the sellers add
    stored energy to their offer — so the evening scarcity (and shortfall) shrinks.
    *Surplus hour* (generation > load, e.g. midday): bank the surplus that would
    otherwise be curtailed — each seller stores its share of its unsold generation
    and the consumer buys a little cheap energy to store.

    Returns ``(off_solar, off_wind, net_demand, flows)`` for the market that hour."""
    gen_total = gs + gw
    surplus = max(0.0, gen_total - load)
    deficit = max(0.0, load - gen_total)
    f = {"charge_solar": 0.0, "discharge_solar": 0.0, "charge_wind": 0.0,
         "discharge_wind": 0.0, "charge_consumer": 0.0, "discharge_consumer": 0.0}
    off_s, off_w, net = gs, gw, load

    if deficit > 1e-9:                                  # evening peak → discharge
        dc = bat_c.dischargeable(deficit); bat_c.discharge(dc)
        net = load - dc; f["discharge_consumer"] = dc
        rem = deficit - dc
        ds = bat_s.dischargeable(rem * 0.5); bat_s.discharge(ds)
        off_s = gs + ds; f["discharge_solar"] = ds
        dw = bat_w.dischargeable(rem - ds); bat_w.discharge(dw)
        off_w = gw + dw; f["discharge_wind"] = dw
    elif surplus > 1e-9:                                # midday surplus → bank it
        share_s = gs / max(gen_total, 1e-9)
        cs = bat_s.chargeable(surplus * share_s); bat_s.charge(cs)
        off_s = gs - cs; f["charge_solar"] = cs
        cw = bat_w.chargeable(surplus * (1.0 - share_s)); bat_w.charge(cw)
        off_w = gw - cw; f["charge_wind"] = cw
        cc = bat_c.chargeable(surplus * 0.3); bat_c.charge(cc)
        net = load + cc; f["charge_consumer"] = cc

    return max(0.0, off_s), max(0.0, off_w), max(0.0, net), f


def run_day(date: str, solar_strategy: str = "opponent_modeling",
            wind_strategy: str = "opponent_modeling",
            buyer_strategy: str = "honest_buyer",
            mean_demand: float | None = None, with_batteries: bool = True,
            rounds: int = 8) -> pd.DataFrame:
    """Roll the negotiation over the 24 hours of ``date``, optionally with a battery
    per agent. Per-hour demand follows an evening-peaked profile so midday is a
    cheap surplus and the evening a dear deficit (the spread storage lives on)."""
    # Hourly generation from the predictive models.
    gens = []
    for h in range(24):
        cfg = scenario_from_timestamp(f"{date} {h:02d}:00:00")
        gens.append((cfg.gen_solar, cfg.gen_wind, cfg.price_min, cfg.price_max,
                     cfg.mc_solar, cfg.mc_wind, cfg.shortfall_penalty))
    gen_total = np.array([gs + gw for gs, gw, *_ in gens])
    if mean_demand is None:
        mean_demand = float(gen_total.mean())
    demand_h = mean_demand * DEFAULT_LOAD_PROFILE

    # One battery per agent, sized to a few hours of peak generation (C/4 power).
    cap = max(1.0, float(gen_total.max()) * 3)
    def new_bat():
        return AgentBattery(capacity=cap, charge_rate=cap / 4, discharge_rate=cap / 4)
    bat_s, bat_w, bat_c = new_bat(), new_bat(), new_bat()

    rows = []
    for h in range(24):
        gs, gw, p_min, p_max, mc_s, mc_w, pen = gens[h]
        load = float(demand_h[h])
        price_fc = _scarcity_price(gs + gw, load, p_min, p_max)  # forecast price signal

        off_s, off_w, net_demand = gs, gw, load
        f = {"charge_solar": 0.0, "discharge_solar": 0.0, "charge_wind": 0.0,
             "discharge_wind": 0.0, "charge_consumer": 0.0, "discharge_consumer": 0.0}
        if with_batteries:
            off_s, off_w, net_demand, f = _schedule_batteries(bat_s, bat_w, bat_c, gs, gw, load)

        cfg_h = MarketConfig(demand=net_demand, gen_solar=off_s, gen_wind=off_w,
                             price_min=p_min, price_max=p_max, mc_solar=mc_s,
                             mc_wind=mc_w, shortfall_penalty=pen,
                             label=f"{date} {h:02d}:00")
        res = run_negotiation(cfg_h, solar_strategy, wind_strategy,
                              buyer_strategy=buyer_strategy, rounds=rounds)
        rows.append({
            "hour": h, "gen_total": round(gs + gw, 3), "load": round(load, 3),
            "net_demand": round(net_demand, 3), "price_fc": round(price_fc, 2),
            "buyer_cost": round(res.buyer_cost, 2), "shortfall": round(res.shortfall, 3),
            "profit_solar": round(res.profit_solar, 2), "profit_wind": round(res.profit_wind, 2),
            "soc_solar": round(bat_s.soc, 3), "soc_wind": round(bat_w.soc, 3),
            "soc_consumer": round(bat_c.soc, 3), **{k: round(v, 3) for k, v in f.items()},
        })
    df = pd.DataFrame(rows)
    df.attrs["capacity"] = cap
    df.attrs["with_batteries"] = with_batteries
    return df


def compare_and_plot(date: str, **kw) -> pd.DataFrame:
    """Run the day with and without per-agent batteries and compare."""
    base = run_day(date, with_batteries=False, **kw)
    batt = run_day(date, with_batteries=True, **kw)

    pen = 1000.0  # shortfall penalty per unmet kW (MarketConfig default)
    # Separate the energy bill from the unmet-demand penalty so the price effect is
    # readable on its own (the penalty otherwise dominates the headline cost).
    base_energy = float((base.buyer_cost - base.shortfall * pen).sum())
    batt_energy = float((batt.buyer_cost - batt.shortfall * pen).sum())
    summary = pd.DataFrame({
        "regime": ["no storage", "per-agent storage"],
        "daily_energy_cost": [base_energy, batt_energy],
        "daily_shortfall_kWh": [base.shortfall.sum(), batt.shortfall.sum()],
        "daily_consumer_cost_total": [base.buyer_cost.sum(), batt.buyer_cost.sum()],
        "daily_profit_solar": [base.profit_solar.sum(), batt.profit_solar.sum()],
        "daily_profit_wind": [base.profit_wind.sum(), batt.profit_wind.sum()],
    }).round(1)

    RESULTS.mkdir(exist_ok=True)
    merged = base.add_suffix("_base").join(batt.add_suffix("_batt"))
    merged.to_csv(RESULTS / "mas_battery_market_day.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5))
    x = batt["hour"]
    cap = batt.attrs["capacity"]
    ax1.plot(x, batt["soc_solar"], marker=".", color="tab:orange", label="SoC solar (AS)")
    ax1.plot(x, batt["soc_wind"], marker=".", color="tab:blue", label="SoC wind (AE)")
    ax1.plot(x, batt["soc_consumer"], marker=".", color="tab:green", label="SoC consumer (AC)")
    ax1.axhline(0.2 * cap, ls="--", color="grey", lw=0.8, label="20–80% band")
    ax1.axhline(0.8 * cap, ls="--", color="grey", lw=0.8)
    ax1.set_ylabel("state of charge (kWh)")
    ax1.set_title(f"Per-agent battery operation over {date} (20–80% SoC band)")
    ax1.legend(fontsize=8, ncol=2); ax1.grid(True, alpha=0.3)

    ax2.plot(x, base["buyer_cost"], marker="o", color="tab:red", ls="--",
             label="hourly consumer cost — no storage")
    ax2.plot(x, batt["buyer_cost"], marker="s", color="tab:green",
             label="hourly consumer cost — per-agent storage")
    axp = ax2.twinx()
    axp.plot(x, batt["price_fc"], color="tab:purple", alpha=0.4, lw=1.0, label="price signal")
    axp.set_ylabel("price signal", color="tab:purple")
    ax2.set_xlabel("hour"); ax2.set_ylabel("consumer cost")
    ax2.set_title("How storage reshapes the daily cost (evening peak shaved)")
    ax2.legend(fontsize=8, loc="upper left"); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "mas_battery_market.png", dpi=130)
    plt.close(fig)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2017-06-15")
    ap.add_argument("--solar", default="opponent_modeling")
    ap.add_argument("--wind", default="opponent_modeling")
    ap.add_argument("--buyer", default="honest_buyer")
    args = ap.parse_args()

    summary = compare_and_plot(args.date, solar_strategy=args.solar,
                               wind_strategy=args.wind, buyer_strategy=args.buyer)
    pd.set_option("display.width", 200)
    print(f"Daily comparison ({args.date}, sellers={args.solar}/{args.wind}, "
          f"consumer={args.buyer}):\n")
    print(summary.to_string(index=False))
    print(f"\nOutputs -> {RESULTS}/ (mas_battery_market_day.csv, mas_battery_market.png)")


if __name__ == "__main__":
    main()
