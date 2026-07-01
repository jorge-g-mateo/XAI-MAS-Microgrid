"""Buyer (consumer) negotiation strategies — M8/M9.

The consumer is no longer a price-taker. Each round it answers the sellers'
``PROPOSE`` with a counter-``PROPOSE`` (a lower price it is willing to pay) and
both sides concede over a few bargaining steps. This is the downward price
pressure that was missing: sellers used to saturate at ``p_max`` because the
buyer had no credible walk-away.

The lever is the **reservation price** = the consumer's value of lost load (the
most it will pay per unit before it prefers to stay unserved). Below the
reservation a deal is struck somewhere between the seller's floor and the buyer's
ceiling; above it the buyer walks (which may cause a shortfall — the trade-off
between a cheaper price and reliability).

Strategies (mirror of the seller side):
  * ``honest_buyer``            reservation = p_max → caps the price but exerts no
                                real downward pull (≈ the old price-taker; default,
                                keeps existing experiments comparable).
  * ``hard_bargainer``          low in-band reservation → pushes the price toward
                                the floor, risking some shortfall.
  * ``opponent_modeling_buyer`` estimates the sellers' floor from how far they
                                concede and targets just above it (M9).
"""

from __future__ import annotations


class BuyerStrategy:
    """Base buyer strategy. Subclasses tune the reservation / opening / concession."""

    name = "base"

    def __init__(self, reservation_ratio: float = 1.0, open_ratio: float = 0.4,
                 raise_rate: float = 0.5):
        # Fractions of the price band [p_min, p_max].
        self.reservation_ratio = reservation_ratio   # 1.0 → reservation = p_max
        self.open_ratio = open_ratio                  # opening counter, low = aggressive
        self.raise_rate = raise_rate                  # how fast the buyer concedes up
        self.observed_floor: float | None = None      # lowest seller price ever seen

    # --- price helpers (over the band) ---------------------------------------
    def reservation(self, p_min: float, p_max: float) -> float:
        return p_min + self.reservation_ratio * (p_max - p_min)

    def opening(self, p_min: float, p_max: float) -> float:
        return p_min + self.open_ratio * (p_max - p_min)

    def raise_counter(self, my_price: float, seller_price: float,
                      p_min: float, p_max: float) -> float:
        """Concede upward toward the seller's price, capped at the reservation."""
        target = my_price + self.raise_rate * (seller_price - my_price)
        return min(target, self.reservation(p_min, p_max))

    def update(self, conceded_prices: list[float]) -> None:
        """Learn from the prices the sellers ended the round at."""
        if conceded_prices:
            lo = min(conceded_prices)
            self.observed_floor = lo if self.observed_floor is None else min(self.observed_floor, lo)


class PriceTakerBuyer(BuyerStrategy):
    """Accepts whatever the sellers ask (opens at the ceiling so it never
    counters). This is the *implicit* buyer of the original model — kept as the
    default so the seller-vs-seller experiments (run_mas, sweep) reproduce exactly,
    and as the baseline against which the bargaining buyers are compared."""

    name = "price_taker"

    def __init__(self):
        super().__init__(reservation_ratio=1.0, open_ratio=1.0, raise_rate=1.0)


class HonestBuyer(BuyerStrategy):
    """Bargains mildly toward a fair mid-band price; reservation at the ceiling so
    it never risks a shortfall, but two-sided concession already pulls the price
    below the seller-saturated ceiling."""

    name = "honest_buyer"

    def __init__(self):
        super().__init__(reservation_ratio=1.0, open_ratio=0.5, raise_rate=0.7)


class HardBargainer(BuyerStrategy):
    """Aggressive low reservation: pushes the price toward the floor and accepts
    some risk of an under-buy (shortfall) rather than overpay."""

    name = "hard_bargainer"

    def __init__(self):
        super().__init__(reservation_ratio=0.2, open_ratio=0.0, raise_rate=0.3)


class OpponentModelingBuyer(BuyerStrategy):
    """Estimates the sellers' floor from how far they concede and sets its
    reservation just above it, extracting surplus without losing supply (M9)."""

    name = "opponent_modeling_buyer"

    def __init__(self, eps_ratio: float = 0.1):
        super().__init__(reservation_ratio=0.6, open_ratio=0.1, raise_rate=0.4)
        self.eps_ratio = eps_ratio

    def reservation(self, p_min: float, p_max: float) -> float:
        # Once we have observed how low sellers go, aim just above that floor;
        # before that, fall back to the static in-band reservation.
        if self.observed_floor is not None:
            eps = self.eps_ratio * (p_max - p_min)
            return min(p_max, max(p_min, self.observed_floor + eps))
        return super().reservation(p_min, p_max)


BUYER_STRATEGIES: dict[str, type[BuyerStrategy]] = {
    s.name: s for s in [PriceTakerBuyer, HonestBuyer, HardBargainer, OpponentModelingBuyer]
}


def make_buyer_strategy(name: str) -> BuyerStrategy:
    if name not in BUYER_STRATEGIES:
        raise KeyError(f"Unknown buyer strategy {name!r}. Options: {list(BUYER_STRATEGIES)}")
    return BUYER_STRATEGIES[name]()
