"""Run a Contract Net negotiation for one market scenario.

Drives several rounds (so adaptive strategies such as opponent modeling can
converge) and reports the final operating point as (profit_solar, profit_wind,
buyer_cost), directly comparable with the optimization Pareto front.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.mas.acl import MessageBus
from src.mas.agents import ConsumerAgent, SellerAgent
from src.mas.buyer_strategies import make_buyer_strategy
from src.mas.strategies import make_strategy
from src.optimization.market import MarketConfig

SOLAR = "AS_solar"
WIND = "AE_wind"
CONSUMER = "AC_consumer"


@dataclass
class NegotiationResult:
    scenario: str
    solar_strategy: str
    wind_strategy: str
    buyer_strategy: str
    profit_solar: float
    profit_wind: float
    buyer_cost: float
    shortfall: float
    rounds: list[dict]
    final_price_solar: float
    final_price_wind: float
    bargain_log: list = None  # per-step buyer/seller concession trace (M8)


def run_negotiation(cfg: MarketConfig, solar_strategy: str, wind_strategy: str,
                    buyer_strategy: str = "price_taker", rounds: int = 12,
                    n_bargain: int = 3, info_level: str = "full",
                    log: bool = False) -> NegotiationResult:
    bus = MessageBus(log=log)

    seller_s = SellerAgent(SOLAR, cfg.gen_solar, cfg.mc_solar,
                           make_strategy(solar_strategy), cfg.price_min, cfg.price_max)
    seller_w = SellerAgent(WIND, cfg.gen_wind, cfg.mc_wind,
                           make_strategy(wind_strategy), cfg.price_min, cfg.price_max)
    consumer = ConsumerAgent(CONSUMER, cfg.demand, bus,
                             make_buyer_strategy(buyer_strategy),
                             cfg.price_min, cfg.price_max,
                             sellers=[SOLAR, WIND], n_bargain=n_bargain,
                             shortfall_penalty=cfg.shortfall_penalty,
                             info_level=info_level)

    for name, ag in [(SOLAR, seller_s), (WIND, seller_w)]:
        bus.register(name, ag.receive)

    history = []
    for r in range(rounds):
        s_rev0, s_del0 = seller_s.revenue, seller_s.delivered
        w_rev0, w_del0 = seller_w.revenue, seller_w.delivered

        rec = consumer.run_round(r)

        # Profit uses the SAME (possibly convex) generation-cost model as
        # clear_market — cost_i(q) = mc_i·q + 0.5·quad_i·q² — so negotiated outcomes
        # are scored on the exact axes of the optimization front (mc=0,quad=0 → the
        # old linear case). q is the quantity delivered in this clearing.
        ds = seller_s.delivered - s_del0
        dw = seller_w.delivered - w_del0
        ps = (seller_s.revenue - s_rev0) - (cfg.mc_solar * ds + 0.5 * cfg.quad_solar * ds * ds)
        pw = (seller_w.revenue - w_rev0) - (cfg.mc_wind * dw + 0.5 * cfg.quad_wind * dw * dw)
        rec.update({
            "profit_solar": ps, "profit_wind": pw,
            "price_solar": seller_s.last_offer.price if seller_s.last_offer else None,
            "price_wind": seller_w.last_offer.price if seller_w.last_offer else None,
        })
        history.append(rec)

    last = history[-1]
    return NegotiationResult(
        scenario=cfg.label,
        solar_strategy=solar_strategy,
        wind_strategy=wind_strategy,
        buyer_strategy=buyer_strategy,
        profit_solar=last["profit_solar"],
        profit_wind=last["profit_wind"],
        buyer_cost=last["cost"],
        shortfall=last["shortfall"],
        rounds=history,
        final_price_solar=last["price_solar"],
        final_price_wind=last["price_wind"],
        bargain_log=list(consumer.bargain_log),
    )


if __name__ == "__main__":
    from src.optimization.market import scenario_from_timestamp

    cfg = scenario_from_timestamp("2017-06-15 19:00:00")
    print(f"Scenario {cfg.label}: demand={cfg.demand:.2f} "
          f"solar={cfg.gen_solar:.2f} wind={cfg.gen_wind:.2f} kW\n")
    res = run_negotiation(cfg, "opponent_modeling", "honest", log=True)
    print(f"\nFINAL: profit_solar={res.profit_solar:.2f} "
          f"profit_wind={res.profit_wind:.2f} buyer_cost={res.buyer_cost:.2f} "
          f"shortfall={res.shortfall:.3f}")
