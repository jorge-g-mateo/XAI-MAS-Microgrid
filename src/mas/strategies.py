"""Seller negotiation strategies for the microgrid market.

Each strategy decides, every round, the **offer** (quantity, price) a seller
makes in response to a CFP, and updates its internal state from the round
feedback. The experiments analyze the impact of:

  * information hiding  -> :class:`InfoHidingStrategy` (withholds true capacity)
  * deception          -> :class:`DeceptionStrategy`  (overstates capacity it
                          cannot deliver)
  * opponent modeling  -> :class:`OpponentModelingStrategy` (estimates the rival's
                          price from feedback and undercuts adaptively)

plus an :class:`HonestStrategy` baseline and a repeated-game
:class:`ReciprocalStrategy` (tit-for-tat: nice, retaliatory, forgiving — M2).
All of them respect the price band [price_min, price_max] from the market config.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Offer:
    quantity: float          # quantity the seller commits to (may differ from real cap)
    price: float             # unit price asked
    deliverable: float       # quantity it can ACTUALLY deliver (<= forecast)


@dataclass
class RoundFeedback:
    """What the consumer tells a seller after clearing a round."""

    sold: float              # how much this seller actually sold
    clearing_price: float    # max accepted price in the market this round
    demand: float            # total demand that round
    won_any: bool            # whether this seller sold anything
    rival_price: float = 0.0  # best (lowest) price a *competitor* cleared at; lets a
                              # reciprocal seller tell apart the rival's price from its own


class Strategy:
    """Base strategy. Subclasses override :meth:`offer`.

    Each strategy also declares the **agent architecture** it embodies (M5),
    following the reactive / deliberative taxonomy of the SMA syllabus:

      * ``reactive``     — a fixed stimulus→response rule; the agent maps the
        current percept (its capacity, the price band) to an action with no
        internal model of the others and no use of history.
      * ``deliberative`` — the agent keeps internal state (a model of the rival,
        a punishment/forgiveness mode) and reasons over the interaction history
        before acting.
    """

    name = "base"
    architecture = "reactive"
    architecture_reason = "fixed stimulus-response rule; no internal model or memory"

    def __init__(self, margin: float = 5.0):
        self.margin = margin
        self.history: list[RoundFeedback] = []

    def offer(self, available: float, mc: float, p_min: float, p_max: float,
              demand: float) -> Offer:
        raise NotImplementedError

    def update(self, fb: RoundFeedback) -> None:
        self.history.append(fb)

    @staticmethod
    def _clip(p: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, p))


class HonestStrategy(Strategy):
    """Reveals full capacity at a small, fixed margin *above the price floor*.

    The margin is band-relative (a fraction of ``[p_min, p_max]``) so it stays
    visible regardless of the band: with ``mc=0`` and a wide regulated band,
    ``mc + margin`` would clip to ``p_min`` and glue the honest seller to the
    floor, leaving the consumer no room to bargain. Pricing a small markup *above*
    the floor keeps honest the cheapest strategy while opening a real two-sided
    bargaining zone ``[floor, ask]`` the consumer can claw back."""

    name = "honest"
    architecture_reason = "fixed rule: a small constant markup above its price floor, full capacity revealed"

    def __init__(self, markup: float = 0.15, **kw):
        super().__init__(**kw)
        self.markup = markup  # small margin as a fraction of the band, above the floor

    def offer(self, available, mc, p_min, p_max, demand) -> Offer:
        floor = max(p_min, mc + self.margin)
        price = self._clip(floor + self.markup * (p_max - p_min), p_min, p_max)
        return Offer(quantity=available, price=price, deliverable=available)


class InfoHidingStrategy(Strategy):
    """Withholds part of its capacity to create artificial scarcity and hold a
    higher price. It still only commits what it can deliver (no failure risk)."""

    name = "info_hiding"
    architecture_reason = "fixed rule: reveal a constant fraction of capacity at a constant markup"

    def __init__(self, reveal_ratio: float = 0.6, markup: float = 0.6, **kw):
        super().__init__(**kw)
        self.reveal_ratio = reveal_ratio
        self.markup = markup  # fraction of the way up to p_max

    def offer(self, available, mc, p_min, p_max, demand) -> Offer:
        q = available * self.reveal_ratio
        price = self._clip(p_min + self.markup * (p_max - p_min), p_min, p_max)
        return Offer(quantity=q, price=price, deliverable=q)


class DeceptionStrategy(Strategy):
    """Overstates the quantity it can deliver to win the contract, then may fail
    to deliver the surplus -> risks leaving the consumer short (the behaviour the
    cooperation requirement warns against)."""

    name = "deception"
    architecture_reason = "fixed rule: overstate deliverable quantity by a constant factor"

    def __init__(self, overstate: float = 1.4, markup: float = 0.5, **kw):
        super().__init__(**kw)
        self.overstate = overstate
        self.markup = markup

    def offer(self, available, mc, p_min, p_max, demand) -> Offer:
        claimed = available * self.overstate           # promises more than it has
        price = self._clip(p_min + self.markup * (p_max - p_min), p_min, p_max)
        return Offer(quantity=claimed, price=price, deliverable=available)


class OpponentModelingStrategy(Strategy):
    """Estimates the rival's price from round feedback and undercuts just enough
    to win the demand it wants, maximizing margin (adaptive tatonnement).

    If it lost demand last round it assumes it was undercut -> lowers its price;
    if it won comfortably it raises its price to extract more surplus. The
    estimate of the rival price is the observed clearing price."""

    name = "opponent_modeling"
    architecture = "deliberative"
    architecture_reason = ("keeps an estimate of the rival's price and reasons over "
                           "round feedback (won/lost, clearing) to adapt its bid")

    def __init__(self, step: float = 4.0, **kw):
        super().__init__(**kw)
        self.step = step
        self.est_rival_price: float | None = None
        self._last_price: float | None = None

    def offer(self, available, mc, p_min, p_max, demand) -> Offer:
        if not self.history:
            price = self._clip(p_max - self.margin, p_min, p_max)  # probe high
        else:
            fb = self.history[-1]
            self.est_rival_price = fb.clearing_price
            if fb.won_any and fb.sold >= min(available, demand) - 1e-6:
                price = (self._last_price or p_max) + self.step      # won all -> push up
            elif fb.won_any:
                price = (self._last_price or fb.clearing_price)      # partial -> hold
            else:
                # lost: undercut the estimated rival price
                target = (self.est_rival_price or p_max) - self.step
                price = max(mc + 0.5, target)
            price = self._clip(price, p_min, p_max)
        self._last_price = price
        return Offer(quantity=available, price=price, deliverable=available)


class ReciprocalStrategy(Strategy):
    """Tit-for-tat / retributive strategy for the repeated market (M2).

    Reciprocity in the repeated Contract Net. The seller is

      * **nice** — it never defects first: it opens cooperatively, holding a
        high price that leaves both sellers a shared surplus;
      * **retaliatory** — if a rival breaks cooperation by undercutting (clears
        the market below the cooperative price and steals the cheap volume), it
        answers with a **price war** for ``punish_rounds`` rounds, dropping to
        the floor to deny the rival its margin;
      * **forgiving** — after the punishment it returns to the cooperative
        price, testing whether the rival will now cooperate.

    This is the textbook mechanism by which **repetition sustains cooperation**:
    the credible threat of retaliation makes unilateral undercutting
    unprofitable. The cooperative price is deliberately *high* — honest pricing
    already sits at the band floor (``mc + margin`` clipped to ``p_min``), so
    there would be no room left to punish below it; here "cooperation" means
    mutual restraint (the tacit collusion both sellers prefer to a price war)
    and "defection" means breaking it by undercutting. This is the repeated-game
    counterpart of the one-shot result (the static Nash equilibrium is mutual
    exploitation; reciprocity is what can hold the cooperative outcome).

    Defection is read from the round feedback: ``won_any == False`` (fully shut
    out) or ``rival_price`` (a competitor's cleared price) falling below the
    cooperative ask. Using the *rival's* price — not the overall market low —
    is what stops the seller from mistaking its own price-war quote for an
    attack while it is still punishing.
    """

    name = "tit_for_tat"
    architecture = "deliberative"
    architecture_reason = ("maintains a cooperate/punish/forgive mode and conditions on "
                           "the rival's past prices (history-dependent reciprocity)")

    def __init__(self, coop_frac: float = 0.7, punish_rounds: int = 2,
                 tol: float = 1.0, **kw):
        super().__init__(**kw)
        self.coop_frac = coop_frac        # cooperative price as a fraction of the band
        self.punish_rounds = punish_rounds
        self.tol = tol
        self.mode = "coop"                # 'coop' | 'war' | 'probe'
        self.punish_left = 0
        self._coop_price: float | None = None
        self._fb: RoundFeedback | None = None   # last round's feedback (latest wins)

    def _defected(self, fb: RoundFeedback | None, coop: float, available: float) -> bool:
        """Did a rival break cooperation last round (undercut the coop price)?"""
        if fb is None or available <= 1e-9:
            return False  # first round, or we had nothing to sell -> no grievance
        if fb.rival_price is None:
            return not fb.won_any  # rival price withheld -> can only react to being shut out
        return (not fb.won_any) or (fb.rival_price + self.tol < coop)

    def offer(self, available, mc, p_min, p_max, demand) -> Offer:
        coop = self._clip(p_min + self.coop_frac * (p_max - p_min), p_min, p_max)
        war = self._clip(mc + 0.5, p_min, p_max)          # price-war floor
        self._coop_price = coop
        fb, self._fb = self._fb, None                     # consume last feedback

        if self.mode == "war":                            # serving out the punishment
            self.punish_left -= 1
            price = war
            if self.punish_left <= 0:
                self.mode = "probe"                       # punishment done -> forgive next
        elif self.mode == "probe":                        # forgiveness: re-cooperate once,
            price = coop                                  # ignoring this round's grievance,
            self.mode = "coop"                            # to test if the rival reciprocates
        else:                                             # 'coop': watch for defection
            if self._defected(fb, coop, available):
                self.mode = "war"
                self.punish_left = self.punish_rounds - 1  # this round is the first war round
                price = war
                if self.punish_left <= 0:
                    self.mode = "probe"
            else:
                price = coop                              # nice: never defect first
        return Offer(quantity=available, price=price, deliverable=available)

    def update(self, fb: RoundFeedback) -> None:
        self.history.append(fb)
        self._fb = fb  # remember latest feedback (robust to >1 update per round)


class BayesianOpponentModelingStrategy(Strategy):
    """Opponent modeling by Bayesian inference over the rival's price (M4).

    Where :class:`OpponentModelingStrategy` is an ad-hoc *tâtonnement* on the
    last clearing price, this strategy *learns* the rival from the **whole
    history of observed offers**, the "estimation by regression / Bayesian
    updating over past offers" the brief asks for. It keeps a conjugate
    **Normal–Normal posterior** over the rival's unit price: a Gaussian belief
    ``N(μ, σ²)`` updated each round with the rival's observed price (the
    ``rival_price`` slot the consumer now reports). Because demand needs both
    sellers, the rival is awarded every round, so its price is observed without
    censoring and the posterior **converges with a shrinking confidence band**.

    Given the belief it **best-responds** using both estimates (price *and*
    capacity), exploiting that demand needs both sellers (cheaper-first, so the
    dearer seller still sells the residual). It compares two regimes and picks
    the higher-revenue one:

      * **undercut** the rival (``μ − margin − λσ``, a confidence-scaled
        undercut: cautious while the belief is diffuse, tightening as σ shrinks)
        → be the cheap seller and move ``min(capacity, demand)`` volume;
      * **over-price** at the ceiling → cede the cheap volume and take the
        *captive residual* ``demand − rival_capacity`` the rival cannot cover,
        at the top price.

    The rival's capacity is tracked as a lower bound (the most it has been seen
    to sell). Exposes ``trace`` (round, μ, σ) for visualization.
    """

    name = "bayesian_opponent"
    architecture = "deliberative"
    architecture_reason = ("keeps a Bayesian posterior (mean + variance) over the rival's "
                           "price from the full offer history and prices a confidence-scaled undercut")

    def __init__(self, margin: float = 2.0, obs_sd: float = 2.0,
                 prior_sd: float | None = None, lam: float = 1.0, **kw):
        super().__init__(**kw)
        self.margin = margin        # base undercut below the estimated rival price
        self.obs_sd = obs_sd        # observation-noise std of an observed rival price
        self.prior_sd = prior_sd    # prior std (None → half the band)
        self.lam = lam              # extra undercut per unit of posterior std (robustness)
        self.mu: float | None = None
        self.var: float | None = None
        self.cap_est: float = 0.0   # lower bound on the rival's capacity (max seen sold)
        self.trace: list[tuple[int, float, float]] = []  # (round, μ, σ)

    def _ensure_prior(self, p_min: float, p_max: float) -> None:
        if self.mu is None:
            self.mu = 0.5 * (p_min + p_max)
            sd = self.prior_sd if self.prior_sd is not None else 0.5 * (p_max - p_min)
            self.var = sd * sd

    def offer(self, available, mc, p_min, p_max, demand) -> Offer:
        self._ensure_prior(p_min, p_max)
        sd = self.var ** 0.5
        floor = max(p_min, mc + 0.5)
        if not self.history:
            price = self._clip(p_max - self.margin, floor, p_max)  # probe high to observe the rival
        else:
            # Best-respond to the belief: pick the higher-revenue of two regimes.
            undercut_p = self._clip(self.mu - self.margin - self.lam * sd, floor, p_max)
            undercut_rev = undercut_p * min(available, demand)      # be the cheap seller
            residual = max(0.0, demand - self.cap_est)              # captive part rival can't cover
            over_p = p_max
            over_rev = over_p * min(available, residual)            # be the dear seller
            price = over_p if over_rev > undercut_rev else undercut_p
        self.trace.append((len(self.history), float(self.mu), float(sd)))
        return Offer(quantity=available, price=price, deliverable=available)

    def update(self, fb: RoundFeedback) -> None:
        self.history.append(fb)
        # Capacity: infer the rival's sales (total served ≈ demand − my sold) → lower bound.
        rival_sold = max(0.0, fb.demand - fb.sold)
        self.cap_est = max(self.cap_est, rival_sold)
        # Price: Normal–Normal conjugate update, but only on an *uncensored* observation —
        # i.e. when the rival was actually awarded, so rival_price is its real price (not
        # the p_max sentinel the consumer reports when no competitor sold).
        if rival_sold > 1e-9 and fb.rival_price is not None:
            x = fb.rival_price
            tau2 = self.obs_sd ** 2
            prec = 1.0 / self.var + 1.0 / tau2
            self.mu = (self.mu / self.var + x / tau2) / prec
            self.var = 1.0 / prec


# The four strategies of the original M3 game (Part I); Part-II additions
# (tit_for_tat, bayesian_opponent) extend the strategy set but the *baseline*
# sub-game is analyzed over exactly these so its documented NE is recoverable.
BASELINE_STRATEGIES: tuple[str, ...] = (
    "honest", "info_hiding", "deception", "opponent_modeling",
)

STRATEGIES: dict[str, type[Strategy]] = {
    s.name: s for s in [HonestStrategy, InfoHidingStrategy,
                        DeceptionStrategy, OpponentModelingStrategy,
                        ReciprocalStrategy, BayesianOpponentModelingStrategy]
}


def make_strategy(name: str, **kw) -> Strategy:
    if name not in STRATEGIES:
        raise KeyError(f"Unknown strategy {name!r}. Options: {list(STRATEGIES)}")
    return STRATEGIES[name](**kw)
