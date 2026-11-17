"""Turn-based engine for the High Society environment.

The engine owns all state transitions.  Agents provide `BidAction` or
`PassAction` values; the engine validates bids and resolves auctions.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import numpy.typing as npt

try:
    from .actions import Action, ActionType, BidAction, PassAction
except ImportError:  # pragma: no cover - supports direct script execution
    from actions import Action, ActionType, BidAction, PassAction


class Player:
    """Storage and helpers for one player's cards."""

    MONEY_CARDS = (1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25)

    def __init__(self) -> None:
        self._money = np.array(self.MONEY_CARDS, dtype=np.int64)
        self._cards: list[str] = []
        self._spent: int = 0
        self._bids: list[int] = []

    @property
    def money_cards(self) -> tuple[int, ...]:
        return tuple(int(card) for card in self._money)

    @property
    def cards(self) -> tuple[str, ...]:
        return tuple(self._cards)

    @property
    def spent(self) -> int:
        return self._spent

    @property
    def bids(self) -> tuple[int, ...]:
        return tuple(self._bids)

    def has_money_card(self, value: int) -> bool:
        """Return whether the player still holds a money card."""
        return bool(np.any(self._money == value))

    def play_money_card(self, value: int) -> None:
        """Discard a held money card and count it as spent."""
        mask = self._money == value
        if not np.any(mask):
            raise ValueError(f"player does not have money card {value}")
        index = int(np.flatnonzero(mask)[0])
        self._money = np.delete(self._money, index)
        self._spent += value
        self._bids.append(value)

    def add_card(self, card: str) -> None:
        """Add an auctioned status card to the player's collection."""
        if card == "b_rem_card":
            numeric_cards = [
                int(name) for name in self._cards if name.isdigit()
            ]
            if numeric_cards:
                highest = str(max(numeric_cards))
                self._cards.remove(highest)
        else:
            self._cards.append(card)

    def calculate_score(self) -> float:
        """Calculate a player's prestige score, including status modifiers."""
        prestige = sum(int(card) for card in self._cards if card.isdigit())

        for _ in range(self._cards.count("b_x2")):
            prestige *= 2
        if "b_x0.5" in self._cards:
            prestige *= 0.5

        return prestige - 5 * self._cards.count("b_neg_5")


class Environment:
    """Manual High Society environment."""

    MAX_SEED = 999_999_999_999
    DECK_CARDS: npt.NDArray[np.str_] = np.array(
        [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "b_x2",
            "b_x2",
            "b_x2",
            "b_rem_card",
            "b_x0.5",
            "b_neg_5",
        ],
        dtype=np.str_,
    )

    def __init__(self, seed: int | None = None, num_players: int = 4) -> None:
        if num_players < 2:
            raise ValueError("High Society requires at least 2 players")
        self.num_players = num_players
        self.reset(seed)

    @property
    def _num_remaining_cards(self) -> int:
        return len(self.DECK_CARDS) - self._shuffle_idx

    @property
    def _remaining_cards(self) -> npt.NDArray[np.str_]:
        return self._deck[self._shuffle_idx :]

    @property
    def _current_card(self) -> str | None:
        if self._shuffle_idx >= len(self._deck):
            return None
        return str(self._deck[self._shuffle_idx])

    @property
    def current_card(self) -> str | None:
        return self._current_card

    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def winner(self) -> int | None:
        return self._winner

    @property
    def eliminated_players(self) -> tuple[int, ...]:
        return self._eliminated_players

    @property
    def auction_active(self) -> bool:
        return self._auction_active

    @property
    def active_players(self) -> tuple[int, ...]:
        return self._active_players

    @property
    def current_bid(self) -> int:
        return self._current_bid

    @property
    def bid_count(self) -> int:
        return self._b_count

    @staticmethod
    def _is_bad_card(card: str) -> bool:
        return card.startswith("b_")

    def start(self, seed: int | None = None) -> None:
        """Reset and start the environment."""
        self.reset(seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset all players and create a fresh shuffled deck."""
        if seed is not None:
            if not isinstance(seed, (int, np.integer)):
                raise TypeError("seed must be an integer or None")
            if seed < 0:
                raise ValueError("seed must be non-negative")
            seed = int(seed)

        rng_seed = (
            random.SystemRandom().randrange(self.MAX_SEED + 1)
            if seed is None
            else seed
        )
        rng = np.random.default_rng(rng_seed)

        self._seed = seed
        self._rng = rng
        self._players = np.array(
            [Player() for _ in range(self.num_players)],
            dtype=object,
        )
        self._deck = np.asarray(rng.permutation(self.DECK_CARDS), dtype=np.str_)
        self._shuffle_idx = 0
        self._b_count = 0
        self._started = True
        self._terminated = False
        self._winner: int | None = None
        self._eliminated_players: tuple[int, ...] = ()

        self._auction_active = False
        self._bad_auction = False
        self._active_players: tuple[int, ...] = ()
        self._passed_players: set[int] = set()
        self._high_bidder: int | None = None
        self._current_bid = 0
        self._starting_player = 0
        self._current_player = 0
        self._last_result: dict[str, Any] | None = None
        return self.get_state()

    def _begin_auction(self) -> None:
        if self._shuffle_idx >= len(self._deck):
            self._finish_game()
            return

        card = str(self._deck[self._shuffle_idx])
        self._auction_active = True
        self._bad_auction = self._is_bad_card(card)
        self._active_players = tuple(range(self.num_players))
        self._passed_players = set()
        self._high_bidder = None
        self._current_bid = 0
        self._current_player = self._starting_player

    def _next_active_player(self) -> None:
        active = set(self._active_players)
        if not active:
            return
        for offset in range(1, self.num_players + 1):
            candidate = (self._current_player + offset) % self.num_players
            if candidate in active:
                self._current_player = candidate
                return

    def _validate_action(self, action: Action) -> None:
        if not isinstance(action, Action):
            raise TypeError("action must be an Action")

        player = self._players[self._current_player]
        if isinstance(action, PassAction):
            return
        if isinstance(action, BidAction):
            if isinstance(action.money_card, (bool, np.bool_)):
                raise ValueError("money_card must be a positive integer")
            if not isinstance(action.money_card, (int, np.integer)):
                raise ValueError("money_card must be a positive integer")
            value = int(action.money_card)
            if value <= 0:
                raise ValueError("money_card must be positive")
            if not player.has_money_card(value):
                raise ValueError(f"player {self._current_player} does not hold money card {value}")
            if value < self._current_bid:
                raise ValueError(
                    f"bid {value} must be at least the current bid {self._current_bid}"
                )
            if not self._bad_auction and value <= self._current_bid:
                raise ValueError(
                    f"bid {value} must be higher than the current bid {self._current_bid}"
                )
            return
        raise TypeError(f"unsupported action type: {type(action).__name__}")

    def step(self, action: Action) -> dict[str, Any]:
        """Process one player's turn and return a serializable step result."""
        if self._terminated:
            raise RuntimeError("the game has ended; call reset() to start again")
        if not self._auction_active:
            self._begin_auction()
            if self._terminated:
                return self.get_state()

        try:
            self._validate_action(action)
        except (TypeError, ValueError):
            if not self._auction_active:
                raise
            self._auction_active = False
            self._bad_auction = False
            self._active_players = ()
            self._passed_players = set()
            self._high_bidder = None
            self._current_bid = 0
            raise
        player_index = self._current_player

        if isinstance(action, PassAction):
            result = self._handle_pass(player_index)
        else:
            result = self._handle_bid(player_index, int(action.money_card)) # type: ignore

        self._last_result = result
        return result

    def _handle_bid(self, player_index: int, value: int) -> dict[str, Any]:
        player = self._players[player_index]
        player.play_money_card(value)
        self._b_count += 1
        self._current_bid = value
        self._high_bidder = player_index

        if not self._bad_auction:
            active = set(self._active_players)
            if len(active) == 1:
                self._resolve_normal_auction(player_index)
                return self.get_state()

        self._next_active_player()
        return self.get_state()

    def _handle_pass(self, player_index: int) -> dict[str, Any]:
        self._passed_players.add(player_index)

        if self._bad_auction:
            self._resolve_bad_auction(player_index)
            return self.get_state()

        active = [
            index
            for index in self._active_players
            if index not in self._passed_players
        ]
        self._active_players = tuple(active)

        if len(active) == 1:
            self._resolve_normal_auction(active[0])
            return self.get_state()

        if active:
            self._next_active_player()
        return self.get_state()

    def _resolve_normal_auction(self, winner: int) -> None:
        card = str(self._deck[self._shuffle_idx])
        self._players[winner].add_card(card)
        self._finish_card(winner)

    def _resolve_bad_auction(self, taker: int) -> None:
        card = str(self._deck[self._shuffle_idx])
        self._players[taker].add_card(card)
        self._finish_card(taker)

    def _finish_card(self, last_winner: int) -> None:
        self._shuffle_idx += 1
        self._auction_active = False
        self._bad_auction = False
        self._active_players = ()
        self._passed_players = set()
        self._high_bidder = None
        self._current_bid = 0
        self._starting_player = (last_winner + 1) % self.num_players
        self._current_player = self._starting_player

        if self._shuffle_idx >= len(self._deck):
            self._finish_game()

    def _finish_game(self) -> None:
        self._terminated = True
        self._auction_active = False

        highest_spend = max(int(player.spent) for player in self._players)
        eliminated = tuple(
            index
            for index, player in enumerate(self._players)
            if int(player.spent) == highest_spend
        )
        self._eliminated_players = eliminated

        eligible = [
            index
            for index in range(self.num_players)
            if index not in eliminated
        ]
        if not eligible:
            eligible = list(range(self.num_players))

        self._winner = max(
            eligible,
            key=lambda index: (
                float(self._players[index].calculate_score()),
                -index,
            ),
        )

    def get_state(self) -> dict[str, Any]:
        """Return a serializable snapshot of the environment."""
        return {
            "seed": self._seed,
            "current_player": self._current_player,
            "current_card": self._current_card,
            "auction_active": self._auction_active,
            "bad_auction": self._bad_auction,
            "current_bid": self._current_bid,
            "high_bidder": self._high_bidder,
            "active_players": list(self._active_players),
            "passed_players": sorted(self._passed_players),
            "remaining_cards": [str(card) for card in self._remaining_cards],
            "terminated": self._terminated,
            "winner": self._winner,
            "eliminated_players": list(self._eliminated_players),
            "players": [
                {
                    "money_cards": list(player.money_cards),
                    "cards": list(player.cards),
                    "spent": player.spent,
                    "score": player.calculate_score(),
                }
                for player in self._players
            ],
        }


__all__ = ["Environment", "Player"]