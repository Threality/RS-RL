"""Action types produced by agents for the High Society environment.

Actions are data objects only. Availability, bidding rules, and state
transitions are handled by the environment engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    """Stable action identifiers suitable for display and serialization."""

    BID = "bid"
    PASS = "pass"


class Action(ABC):
    """Base type for all High Society actions."""

    @property
    @abstractmethod
    def kind(self) -> ActionType:
        """Return the action's stable identifier."""
        ...

    @property
    def action_type(self) -> ActionType:
        """Alias for :attr:`kind`."""
        return self.kind


@dataclass(frozen=True, slots=True)
class BidAction(Action):
    """Bid by playing one money card."""

    money_card: int

    @property
    def kind(self) -> ActionType:
        return ActionType.BID

    @property
    def card(self) -> int:
        """Alias for :attr:`money_card`."""
        return self.money_card

    @property
    def value(self) -> int:
        """Alias for :attr:`money_card`."""
        return self.money_card


@dataclass(frozen=True, slots=True)
class PassAction(Action):
    """Withdraw from the current auction or refuse to bid."""

    @property
    def kind(self) -> ActionType:
        return ActionType.PASS


# Concise aliases for callers that prefer domain-style class names.
Bid = BidAction
Pass = PassAction


__all__ = [
    "Action",
    "ActionType",
    "Bid",
    "BidAction",
    "Pass",
    "PassAction",
]