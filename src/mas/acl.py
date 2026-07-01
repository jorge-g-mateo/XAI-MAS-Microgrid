"""Minimal FIPA-ACL messaging layer (in-process).

Implements the standard FIPA-ACL message structure and a synchronous message bus
so the agents communicate with real ACL semantics (performatives, conversation
ids, protocol, reply chaining) without needing an external XMPP server. The set
of performatives and message slots follows the FIPA ACL specification
(SC00037J / SC00061G).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class Performative(str, Enum):
    """Subset of FIPA-ACL communicative acts used by the microgrid protocol."""

    CFP = "cfp"                          # call for proposals (initiator -> sellers)
    PROPOSE = "propose"                  # a bid (seller -> initiator)
    ACCEPT_PROPOSAL = "accept-proposal"  # bid accepted (initiator -> seller)
    REJECT_PROPOSAL = "reject-proposal"  # bid rejected (initiator -> seller)
    REFUSE = "refuse"                    # seller declines to bid
    INFORM = "inform"                    # round result feedback
    FAILURE = "failure"                  # a seller failed to deliver


_counter = itertools.count(1)


@dataclass
class ACLMessage:
    """A FIPA-ACL message. ``content`` carries the domain payload (a dict)."""

    performative: Performative
    sender: str
    receiver: str
    content: dict[str, Any] = field(default_factory=dict)
    conversation_id: str = ""
    protocol: str = "fipa-contract-net"
    reply_with: str = ""
    in_reply_to: str = ""
    language: str = "json"
    ontology: str = "microgrid-energy-market"

    def __post_init__(self):
        if not self.reply_with:
            self.reply_with = f"m{next(_counter)}"

    def reply(self, performative: Performative, content: dict | None = None) -> "ACLMessage":
        """Build a reply that preserves the conversation and reply chain."""
        return ACLMessage(
            performative=performative,
            sender=self.receiver,
            receiver=self.sender,
            content=content or {},
            conversation_id=self.conversation_id,
            protocol=self.protocol,
            in_reply_to=self.reply_with,
        )

    def __repr__(self) -> str:
        return (f"ACL[{self.performative.value}] {self.sender}->{self.receiver} "
                f"{self.content}")


class MessageBus:
    """Synchronous in-process router: delivers a message and returns the reply.

    When ``validate`` is set (default), every message's content is checked
    against the explicit market ontology (:mod:`src.mas.ontology`) before it is
    delivered, so malformed messages fail loudly instead of silently — the
    ontology is *enforced* by the transport, not merely advertised in the
    ``ontology`` slot.
    """

    def __init__(self, log: bool = False, validate: bool = True):
        self._handlers: dict[str, Callable[[ACLMessage], ACLMessage | None]] = {}
        self.transcript: list[ACLMessage] = []
        self.log = log
        self.validate = validate

    def register(self, name: str, handler: Callable[[ACLMessage], ACLMessage | None]) -> None:
        self._handlers[name] = handler

    def _check(self, msg: ACLMessage) -> None:
        # Imported lazily to keep acl.py free of any import cycle.
        from src.mas.ontology import validate_content
        validate_content(msg.performative.value, msg.content)

    def send(self, msg: ACLMessage) -> ACLMessage | None:
        """Deliver ``msg`` to its receiver and return that agent's reply (if any)."""
        if self.validate:
            self._check(msg)
        self.transcript.append(msg)
        if self.log:
            print("  ", msg)
        reply = self._handlers[msg.receiver](msg)
        if reply is not None:
            if self.validate:
                self._check(reply)
            self.transcript.append(reply)
            if self.log:
                print("  ", reply)
        return reply
