"""Microgrid agents: two renewable sellers (AS, AE) and a consumer (AC).

Communication uses the FIPA-ACL layer (:mod:`src.mas.acl`) and the Contract Net
interaction protocol. The consumer is the initiator; the sellers are the
participants.

Market clearing is *equivalent* to the canonical :func:`clear_market` rule from
the optimization module — cheaper offer first, capped by demand — but it is
reimplemented here as a message exchange (CFP -> PROPOSE -> ACCEPT/REJECT ->
INFORM) rather than a single pure call, because the negotiation needs the
multi-round protocol and must model real deliveries. For the honest case the two
agree exactly, so the negotiated point is directly comparable with the Pareto
front; the deception strategy deliberately *diverges* (a seller may promise more
than it can deliver, which :func:`clear_market` cannot express) — that gap is the
phenomenon being studied, not an inconsistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.mas.acl import ACLMessage, MessageBus, Performative
from src.mas.buyer_strategies import BuyerStrategy
from src.mas.strategies import Offer, RoundFeedback, Strategy


@dataclass
class SellerAgent:
    """A generator agent (solar or wind). Responds to CFPs with a PROPOSE."""

    name: str
    available: float          # forecasted generation this scenario (kW), the hard cap
    marginal_cost: float
    strategy: Strategy
    p_min: float
    p_max: float
    min_margin: float = 0.0   # floor above marginal cost it will not concede below
    concession: float = 0.5   # fraction of the gap it gives up per bargaining step
    min_capacity: float = 1e-3  # below this (kW) the seller has nothing to sell -> REFUSE
    # bookkeeping
    last_offer: Offer | None = None
    revenue: float = 0.0
    delivered: float = 0.0
    failures: int = 0
    refusals: int = 0

    @property
    def floor(self) -> float:
        """Lowest price the seller will accept (price band ∧ cost + min margin)."""
        return max(self.p_min, self.marginal_cost + self.min_margin)

    def receive(self, msg: ACLMessage) -> ACLMessage | None:
        if msg.performative == Performative.CFP:
            demand = msg.content["demand"]
            # Capability check before consulting the pricing strategy: a seller
            # with no generation to sell, or whose price floor sits above the
            # band, cannot participate and answers with a FIPA REFUSE (M5).
            if self.available <= self.min_capacity:
                self.refusals += 1
                self.last_offer = None
                return msg.reply(Performative.REFUSE, {"reason": "no_capacity"})
            if self.floor > self.p_max + 1e-9:
                self.refusals += 1
                self.last_offer = None
                return msg.reply(Performative.REFUSE, {"reason": "floor_above_band"})
            offer = self.strategy.offer(
                self.available, self.marginal_cost, self.p_min, self.p_max, demand)
            self.last_offer = offer
            return msg.reply(Performative.PROPOSE,
                             {"quantity": offer.quantity, "price": offer.price})

        if msg.performative == Performative.PROPOSE:  # buyer counter-offer (bargaining)
            counter = msg.content["price"]
            cur = self.last_offer.price if self.last_offer else self.p_max
            # Concede partway toward the counter, never below the floor.
            new_price = max(self.floor, cur - self.concession * (cur - counter))
            qty = self.last_offer.quantity if self.last_offer else 0.0
            deliv = self.last_offer.deliverable if self.last_offer else 0.0
            self.last_offer = Offer(quantity=qty, price=new_price, deliverable=deliv)
            return msg.reply(Performative.PROPOSE, {"quantity": qty, "price": new_price})

        if msg.performative == Performative.ACCEPT_PROPOSAL:
            awarded = msg.content["quantity"]
            price = msg.content["price"]
            # A seller can only physically deliver up to `deliverable`.
            deliver = min(awarded, self.last_offer.deliverable)
            shortfall = awarded - deliver
            self.delivered += deliver
            self.revenue += deliver * price
            if shortfall > 1e-6:
                self.failures += 1
                return msg.reply(Performative.FAILURE,
                                 {"delivered": deliver, "missing": shortfall})
            return msg.reply(Performative.INFORM, {"delivered": deliver})

        if msg.performative == Performative.INFORM:  # round feedback
            self.strategy.update(RoundFeedback(**msg.content))
            return None

        if msg.performative == Performative.REJECT_PROPOSAL:
            self.strategy.update(RoundFeedback(
                sold=0.0, clearing_price=msg.content.get("clearing_price", 0.0),
                demand=msg.content.get("demand", 0.0), won_any=False))
            return None
        return None

    @property
    def profit(self) -> float:
        return self.revenue - self.marginal_cost * self.delivered


@dataclass
class ConsumerAgent:
    """The buyer (AC). Runs the Contract Net as initiator and **bargains**: it
    answers each seller's PROPOSE with a counter-PROPOSE and both sides concede
    over ``n_bargain`` steps, up to the buyer's reservation price (M8/M9)."""

    name: str
    demand: float
    bus: MessageBus
    strategy: BuyerStrategy
    price_min: float
    price_max: float
    sellers: list[str] = field(default_factory=list)
    n_bargain: int = 3
    total_cost: float = 0.0
    total_received: float = 0.0
    shortfall_penalty: float = 1000.0
    info_level: str = "full"  # feedback richness (M6): full | clearing | blind
    bargain_log: list = field(default_factory=list)  # per-step concession trace (M8)

    def run_round(self, round_id: int) -> dict:
        """One round: CFP -> bargain (counter-offers) -> award cheaper first."""
        conv = f"cnp-r{round_id}"
        p_min, p_max = self.price_min, self.price_max

        # 1) CFP -> initial quotes.
        quotes: dict[str, dict] = {}
        for s in self.sellers:
            reply = self.bus.send(ACLMessage(Performative.CFP, self.name, s,
                                             {"demand": self.demand}, conversation_id=conv))
            if reply and reply.performative == Performative.PROPOSE:
                quotes[s] = dict(reply.content)

        # 2) Bilateral bargaining: the buyer counters low and raises toward the
        #    seller's price (capped at its reservation); the seller concedes down.
        buyer_price = {s: self.strategy.opening(p_min, p_max) for s in quotes}
        for s in quotes:  # step 0: opening positions, before any concession
            self.bargain_log.append({"round": round_id, "step": 0, "seller": s,
                                     "buyer_price": buyer_price[s],
                                     "seller_price": quotes[s]["price"]})
        for step in range(self.n_bargain):
            for s in list(quotes):
                if buyer_price[s] >= quotes[s]["price"] - 1e-9:
                    self.bargain_log.append({"round": round_id, "step": step + 1, "seller": s,
                                             "buyer_price": buyer_price[s],
                                             "seller_price": quotes[s]["price"]})
                    continue  # already agreeable
                resp = self.bus.send(ACLMessage(
                    Performative.PROPOSE, self.name, s,
                    {"price": buyer_price[s], "quantity": quotes[s]["quantity"]},
                    conversation_id=conv))
                if resp and resp.performative == Performative.PROPOSE:
                    quotes[s]["price"] = resp.content["price"]
                    quotes[s]["quantity"] = resp.content.get("quantity", quotes[s]["quantity"])
                buyer_price[s] = self.strategy.raise_counter(
                    buyer_price[s], quotes[s]["price"], p_min, p_max)
                self.bargain_log.append({"round": round_id, "step": step + 1, "seller": s,
                                         "buyer_price": buyer_price[s],
                                         "seller_price": quotes[s]["price"]})

        # 3) A seller is acceptable only if its final price is within reservation.
        reservation = self.strategy.reservation(p_min, p_max)
        deals = {s: q for s, q in quotes.items() if q["price"] <= reservation + 1e-9}

        # 4) Award: cheaper price first, up to demand and offered quantity.
        remaining = self.demand
        awards: dict[str, float] = {}
        clearing_price = 0.0
        for s, p in sorted(deals.items(), key=lambda kv: kv[1]["price"]):
            take = max(0.0, min(p["quantity"], remaining))
            awards[s] = take
            if take > 0:
                clearing_price = max(clearing_price, p["price"])
                remaining -= take

        # 5) ACCEPT / REJECT, track real deliveries (deception may cause failures).
        received = 0.0
        cost = 0.0
        for s, p in quotes.items():
            take = awards.get(s, 0.0)
            if take > 0:
                resp = self.bus.send(ACLMessage(Performative.ACCEPT_PROPOSAL, self.name, s,
                                                {"quantity": take, "price": p["price"]},
                                                conversation_id=conv))
                delivered = resp.content.get("delivered", take) if resp else take
                received += delivered
                cost += delivered * p["price"]
            else:
                self.bus.send(ACLMessage(Performative.REJECT_PROPOSAL, self.name, s,
                                         {"clearing_price": None if self.info_level == "blind"
                                          else clearing_price,
                                          "demand": self.demand},
                                         conversation_id=conv))

        # 6) INFORM feedback (enables seller-side opponent modeling next round).
        #    Each seller is told the best (lowest) price a *competitor* cleared
        #    at, so a reciprocal seller can tell a rival's undercut apart from its
        #    own price; p_max means no competitor sold (it was not undercut).
        for s, p in quotes.items():
            sold = awards.get(s, 0.0)
            rival_lows = [q["price"] for s2, q in quotes.items()
                          if s2 != s and awards.get(s2, 0.0) > 0]
            rival_price = min(rival_lows) if rival_lows else p_max
            # Value-of-information ablation (M6): the consumer may withhold price
            # signals. 'clearing' hides the rival's price; 'blind' hides all prices.
            shown_clearing = None if self.info_level == "blind" else clearing_price
            shown_rival = rival_price if self.info_level == "full" else None
            self.bus.send(ACLMessage(Performative.INFORM, self.name, s,
                                     {"sold": sold, "clearing_price": shown_clearing,
                                      "demand": self.demand, "won_any": sold > 0,
                                      "rival_price": shown_rival},
                                     conversation_id=conv))

        # 7) The buyer learns how low the sellers went (M9 opponent modeling).
        self.strategy.update([q["price"] for q in quotes.values()])

        shortfall = max(0.0, self.demand - received)
        cost += shortfall * self.shortfall_penalty
        self.total_cost += cost
        self.total_received += received
        return {"round": round_id, "received": received, "cost": cost,
                "shortfall": shortfall, "clearing_price": clearing_price,
                "awards": awards, "reservation": reservation}
