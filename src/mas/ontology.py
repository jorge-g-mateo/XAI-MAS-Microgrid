"""Explicit ontology and content schema for the microgrid market (M5).

The FIPA-ACL layer (:mod:`src.mas.acl`) already tags every message with
``ontology="microgrid-energy-market"``, but that ontology lived *implicitly* in
the payload dicts. This module makes it **explicit**: it names the domain
**concepts**, the **slots** of each concept, and the **content schema** every
performative must satisfy. :class:`~src.mas.acl.MessageBus` validates messages
against this schema, so the ontology is *enforced*, not just documented.

It also records the **agent-architecture taxonomy** (reactive vs deliberative)
of the negotiation strategies, read directly from the strategy classes so the
documentation cannot drift from the code.

This keeps the module dependency-light (the schema is keyed by the performative
*string*, so there is no import cycle with :mod:`src.mas.acl`).
"""

from __future__ import annotations

ONTOLOGY_NAME = "microgrid-energy-market"

# --- Domain concepts: the vocabulary of the market and the slots each carries.
CONCEPTS: dict[str, dict[str, str]] = {
    "Demand": {
        "demand": "energy the consumer needs this round (kW)",
    },
    "Offer": {
        "quantity": "quantity the seller commits to sell (kW)",
        "price": "unit price asked (within the price band)",
    },
    "Award": {
        "quantity": "quantity the consumer accepts from this seller (kW)",
        "price": "agreed unit price",
    },
    "ClearingResult": {
        "sold": "quantity this seller actually sold (kW)",
        "clearing_price": "highest accepted price in the market this round",
        "rival_price": "best (lowest) price a competitor cleared at",
        "demand": "total demand that round (kW)",
        "won_any": "whether this seller sold anything",
    },
    "Delivery": {
        "delivered": "quantity actually delivered (kW)",
        "missing": "quantity promised but not delivered (kW)",
    },
    "Refusal": {
        "reason": "why the seller declines to bid (e.g. no_capacity)",
    },
}

# --- Content schema: the slots each performative's content MUST contain.
#     Each performative maps to a list of allowed content *shapes* (alternatives);
#     a message is valid if it satisfies ANY shape. INFORM is overloaded (as in
#     FIPA, it just asserts a proposition): a seller's delivery report vs. the
#     consumer's end-of-round clearing result. Extra slots are always tolerated.
MESSAGE_SCHEMA: dict[str, list[set[str]]] = {
    "cfp": [{"demand"}],                                   # Demand
    "propose": [{"quantity", "price"}],                    # Offer (or buyer counter-Offer)
    "accept-proposal": [{"quantity", "price"}],            # Award
    "reject-proposal": [{"clearing_price", "demand"}],     # partial ClearingResult
    "inform": [{"delivered"},                              # Delivery report (seller -> AC)
               {"sold", "clearing_price", "rival_price",   # ClearingResult (AC -> seller)
                "demand", "won_any"}],
    "failure": [{"delivered", "missing"}],                 # Delivery (shortfall)
    "refuse": [{"reason"}],                                # Refusal
}


class OntologyError(ValueError):
    """Raised when a message's content violates the market ontology."""


def validate_content(performative: str, content: dict) -> None:
    """Check that ``content`` matches one of the ontology shapes for the act.

    Raises :class:`OntologyError` if the performative is unknown or the content
    satisfies none of the allowed shapes. Extra slots are tolerated (the
    ontology is open).
    """
    if performative not in MESSAGE_SCHEMA:
        raise OntologyError(f"performative {performative!r} is not in the "
                            f"microgrid ontology {ONTOLOGY_NAME!r}")
    shapes = MESSAGE_SCHEMA[performative]
    have = set(content)
    if any(required <= have for required in shapes):
        return
    allowed = " | ".join("{" + ", ".join(sorted(s)) + "}" for s in shapes)
    raise OntologyError(
        f"{performative!r} message content {sorted(have)} matches no ontology "
        f"shape (allowed: {allowed})")


def architecture_taxonomy() -> dict[str, tuple[str, str]]:
    """Map each negotiation strategy to its (architecture, reason).

    Read from the strategy classes (lazy import to avoid an import cycle) so this
    taxonomy stays in sync with the code.
    """
    from src.mas.strategies import STRATEGIES
    return {name: (cls.architecture, cls.architecture_reason)
            for name, cls in STRATEGIES.items()}


def summary() -> str:
    """A human-readable dump of the ontology + architecture taxonomy."""
    lines = [f"Ontology: {ONTOLOGY_NAME}", "", "Concepts:"]
    for concept, slots in CONCEPTS.items():
        lines.append(f"  {concept}")
        for slot, desc in slots.items():
            lines.append(f"    - {slot}: {desc}")
    lines += ["", "Message content schema (required slots; '|' = alternative shapes):"]
    for perf, shapes in MESSAGE_SCHEMA.items():
        shown = " | ".join("{" + ", ".join(sorted(s)) + "}" for s in shapes)
        lines.append(f"  {perf:<16} {shown}")
    lines += ["", "Agent architecture taxonomy:"]
    for name, (arch, why) in architecture_taxonomy().items():
        lines.append(f"  {name:<18} {arch:<13} {why}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
