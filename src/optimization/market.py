"""Energy market model shared by the optimization and the multi-agent modules.

This implements the shared market contract: two sellers (Solar = AS, Wind = AE)
offer a quantity and a price, and a buyer (Consumer = AC) clears the market
buying the cheaper offer first. The single, pure
``clear_market`` function is the canonical market-clearing rule so that BOTH the
NSGA optimization and the FIPA-ACL negotiation evaluate outcomes identically.

The key integration point with the predictive models: the *maximum* quantity each
seller can offer is its forecasted generation, obtained from
``src/common/inference.py``. Build a scenario with :func:`scenario_from_timestamp`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.common.inference import (
    load_solar_data,
    load_wind_data,
    predict_solar,
    predict_wind,
)

# --- Unit harmonization ------------------------------------------------------
# The two target columns are NOT in the same unit system:
#   * Solar `SystemProduction` is absolute energy in Wh per hourly record
#     (~ average watts). Dividing by 1000 gives kW.
#   * Wind `Power` is the % of the turbine's rated capacity (0-100), evidenced by
#     its saturation at ~100 regardless of wind speed. Converting to kW requires
#     a turbine rated power: kW = Power/100 * rated_kw.
# We express the whole market in kW so both agents are physically comparable.
SOLAR_WH_TO_KW = 1.0 / 1000.0
DEFAULT_WIND_RATED_KW = 10.0  # assumed single-turbine rating (tunable)


@dataclass
class MarketConfig:
    """All numeric parameters of one market instance (one timestep).

    ``gen_solar``/``gen_wind`` are the forecasted generations (upper bounds on the
    quantity each seller may offer) and come from the predictive models.
    """

    demand: float                 # energy the consumer needs (kW)
    gen_solar: float              # max solar power available (model forecast)
    gen_wind: float               # max wind power available (model forecast)
    price_min: float = 50.0       # price domain lower bound
    price_max: float = 80.0       # price domain upper bound
    mc_solar: float = 0.0         # linear marginal-cost coefficient, solar seller
    mc_wind: float = 0.0          # linear marginal-cost coefficient, wind seller
    # Quadratic (convex) generation-cost coefficients. With these at 0 the costs are
    # purely linear (mc=0 → the degenerate plane where C_AC = π_s + π_w identically).
    # Positive values make each generator's cost convex in its output (economic
    # dispatch), so C_AC = π_s + π_w + G(q) with G convex → the front gains real
    # curvature and a meaningful knee. cost_i(q) = mc_i·q + 0.5·quad_i·q²;
    # marginal cost = mc_i + quad_i·q.
    quad_solar: float = 0.0
    quad_wind: float = 0.0
    shortfall_penalty: float = 1000.0  # penalty per unit of unmet demand
    enable_battery: bool = False  # if True, a battery may buy the unsold surplus
    battery_cap: float = 1e9      # max quantity the battery may buy (kW this step)
    label: str = ""               # human-readable scenario tag

    # convenience: stored context (timestamp, etc.) for reporting
    meta: dict = field(default_factory=dict)


@dataclass
class MarketOutcome:
    """Result of clearing the market for a given set of offers."""

    sold_solar: float
    sold_wind: float
    profit_solar: float
    profit_wind: float
    buyer_cost: float       # money paid by the consumer (incl. penalty)
    shortfall: float        # unmet demand
    battery_bought: float = 0.0   # surplus the battery bought (kW)
    battery_profit: float = 0.0   # battery's markup profit (proxy for temporal spread)


def clear_market(
    q_solar: float,
    q_wind: float,
    p_solar: float,
    p_wind: float,
    cfg: MarketConfig,
    q_batt: float = 0.0,
    markup: float = 1.0,
) -> MarketOutcome:
    """Clear the market for one set of offers (the canonical rule).

    The buyer covers its demand starting from the cheaper offer, limited by each
    seller's offered quantity (which is itself capped at its forecasted
    generation). Returns profits, buyer cost and any shortfall.

    Battery (optional, ``cfg.enable_battery``): after the consumer is served, the
    battery may buy up to ``q_batt`` of the **unsold surplus** (cheaper seller
    first) and resells it at ``markup × buy_price``. Its profit
    ``= bought × buy_price × (markup − 1)`` is a **proxy for the temporal spread**
    (it monetizes surplus that would otherwise be curtailed). Battery purchases add
    to the sellers' revenue; the consumer is served first, so it is never harmed.
    """
    # Offers cannot exceed forecasted generation (resource scarcity).
    q_solar = max(0.0, min(q_solar, cfg.gen_solar))
    q_wind = max(0.0, min(q_wind, cfg.gen_wind))

    demand = cfg.demand

    # Buyer prefers the cheaper seller first.
    if p_solar <= p_wind:
        sold_solar = min(q_solar, demand)
        sold_wind = min(q_wind, demand - sold_solar)
    else:
        sold_wind = min(q_wind, demand)
        sold_solar = min(q_solar, demand - sold_wind)

    shortfall = max(0.0, demand - (sold_solar + sold_wind))

    # Consumer pays only for the energy it takes (+ shortfall penalty).
    buyer_cost = sold_solar * p_solar + sold_wind * p_wind + cfg.shortfall_penalty * shortfall

    # Battery buys the unsold surplus (cheaper seller first), reselling at a markup.
    batt_solar = batt_wind = battery_profit = 0.0
    if cfg.enable_battery and q_batt > 1e-12 and markup > 1.0:
        want = min(q_batt, cfg.battery_cap)
        # cheaper seller's leftover first
        order = ([("solar", q_solar - sold_solar, p_solar), ("wind", q_wind - sold_wind, p_wind)]
                 if p_solar <= p_wind else
                 [("wind", q_wind - sold_wind, p_wind), ("solar", q_solar - sold_solar, p_solar)])
        for who, avail, price in order:
            take = max(0.0, min(avail, want))
            battery_profit += take * price * (markup - 1.0)
            if who == "solar":
                batt_solar = take
            else:
                batt_wind = take
            want -= take

    # Battery purchases add to the sellers' revenue. Generation cost is convex:
    # cost_i(q) = mc_i·q + 0.5·quad_i·q² over the total produced (sold + stored).
    prod_solar = sold_solar + batt_solar
    prod_wind = sold_wind + batt_wind
    cost_solar = cfg.mc_solar * prod_solar + 0.5 * cfg.quad_solar * prod_solar * prod_solar
    cost_wind = cfg.mc_wind * prod_wind + 0.5 * cfg.quad_wind * prod_wind * prod_wind
    profit_solar = prod_solar * p_solar - cost_solar
    profit_wind = prod_wind * p_wind - cost_wind

    return MarketOutcome(
        sold_solar=sold_solar,
        sold_wind=sold_wind,
        profit_solar=profit_solar,
        profit_wind=profit_wind,
        buyer_cost=buyer_cost,
        shortfall=shortfall,
        battery_bought=batt_solar + batt_wind,
        battery_profit=battery_profit,
    )


def scenario_from_timestamp(
    timestamp: str | int,
    demand: float | None = None,
    demand_ratio: float = 0.8,
    wind_rated_kw: float = DEFAULT_WIND_RATED_KW,
    **cfg_overrides,
) -> MarketConfig:
    """Build a :class:`MarketConfig` for a real timestep using model forecasts.

    ``timestamp`` can be a ``Date`` string present in the datasets or an integer
    row index. ``gen_solar``/``gen_wind`` come from the model predictions for that
    row, **converted to a common unit (kW)** (see module-level note on units).
    If ``demand`` is not given, it defaults to ``demand_ratio`` times the total
    available generation (so the sellers are genuinely scarce and must negotiate /
    cooperate).
    """
    solar_df = load_solar_data()
    wind_df = load_wind_data()

    if isinstance(timestamp, int):
        solar_row = solar_df.iloc[[timestamp]]
        wind_row = wind_df.iloc[[timestamp]]
        tag = f"row={timestamp}"
    else:
        solar_row = solar_df[solar_df["Date"] == timestamp]
        wind_row = wind_df[wind_df["Date"] == timestamp]
        if solar_row.empty or wind_row.empty:
            raise KeyError(f"Timestamp {timestamp!r} not found in both datasets.")
        tag = str(timestamp)

    gen_solar_raw = max(0.0, float(predict_solar(solar_row)[0]))   # Wh
    gen_wind_raw = max(0.0, float(predict_wind(wind_row)[0]))      # % of rated

    # Convert to a common physical unit (kW).
    gen_solar = gen_solar_raw * SOLAR_WH_TO_KW
    gen_wind = (gen_wind_raw / 100.0) * wind_rated_kw

    if demand is None:
        demand = demand_ratio * (gen_solar + gen_wind)

    return MarketConfig(
        demand=demand,
        gen_solar=gen_solar,
        gen_wind=gen_wind,
        label=tag,
        meta={
            "timestamp": tag,
            "gen_solar_raw_Wh": gen_solar_raw,
            "gen_wind_raw_pct": gen_wind_raw,
            "wind_rated_kw": wind_rated_kw,
            "unit": "kW",
        },
        **cfg_overrides,
    )


# Economic-dispatch cost preset that turns the degenerate plane into a curved
# front. Solar is cheap at low output but steeply convex (costly to push to its
# small capacity); wind has a higher base but a flatter curve — an asymmetry that
# makes the least-cost dispatch a genuine, non-trivial trade-off. Marginal cost
# mc_i + quad_i·q stays inside the price band over the operating range, so sellers
# remain profitable and the feasible region is well populated.
CURVED_COSTS = dict(mc_solar=15.0, quad_solar=20.0, mc_wind=40.0, quad_wind=8.0)


def scenario_curved(timestamp: str | int, freedom_ratio: float = 0.9,
                    **overrides) -> MarketConfig:
    """Build a scenario whose Pareto front is genuinely **curved** (not the
    degenerate plane). Two ingredients are required and both are supplied here:

      1. **Convex asymmetric generation costs** (``CURVED_COSTS``) so that
         ``C_AC = π_s + π_w + G(q)`` with ``G`` convex — breaking the linear
         dependence that makes the mc=0 front a flat plane.
      2. **Dispatch freedom**: demand is set to ``freedom_ratio · min(gen_s,
         gen_w)`` so either generator could nearly cover it alone and the dispatch
         split can vary over its full range. Without this the split is forced
         (a scarce hour where demand exceeds one generator) and the front stays
         nearly flat even with convex costs.

    The mc=0 plane (``scenario_from_timestamp``) is kept as the degenerate baseline;
    this is the curved counterpart for the same machinery and indicators.
    """
    base = scenario_from_timestamp(timestamp)
    demand = freedom_ratio * min(base.gen_solar, base.gen_wind)
    costs = {**CURVED_COSTS, **overrides}
    return scenario_from_timestamp(timestamp, demand=demand, **costs)


if __name__ == "__main__":
    cfg = scenario_from_timestamp("2017-06-15 13:00:00")
    print("Scenario:", cfg)
    out = clear_market(cfg.gen_solar, cfg.gen_wind, 60, 70, cfg)
    print("Outcome:", out)
    cfg2 = scenario_curved("2017-06-15 13:00:00")
    print("Curved scenario:", cfg2)
