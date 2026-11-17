from random import randint
from actions import Action
import pandas as pd
import numpy as np
import numpy.typing as npt

class Environment:
    MAX_SEED = 999_999_999_999
    DECK_CARDS: npt.NDArray[np.str_] = np.array(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'b_x2', 'b_x2', 'b_x2', 'b_rem_card', 'b_x0.5', 'b_neg_5'])
    """
    High Society Environment

    This environment serves as one of my testing environments. It is generalised to be interacted with by the generic program. It should be possible to replace this environment with any other as long as the high level functions remain standardised.

    This environment specifically is an implementation of the card game high society where players bid for cards using a fixed set of money cards. The player with the highest amount of points at the end wins, however the person who spent the most money is eliminated.

    To run with a set seed call reset with seed. __init__ uses a random seed.
    """
    def __init__(self) -> None:
        """
        Prepares an instance to be played on
        """
        self.reset()

    @property
    def _num_remaining_cards(self):
        """
        Number of cards still in the deck
        """
        return len(self.DECK_CARDS) - self._shuffle_idx

    @property
    def _remaining_cards(self):
        """
        Get a list of cards still in the deck.
        """
        return self._deck[self._shuffle_idx:]

    @property
    def _current_card(self):
        """
        Gets the card currently on the table.
        """
        return self._deck[self._shuffle_idx]

    def start(self, seed=randint(0,MAX_SEED)) -> None:
        """
        Starts a game environment.
        """

    def reset(self, seed=randint(0,MAX_SEED)) -> None:
        """
        Randomises starting values and preps a game of High Society to be played.
        """
        self._players: npt.NDArray[np.object_] = np.array(
            [Player() for _ in range(4)], dtype=Player
        )
        self._deck: npt.NDArray[np.str_] = np.random.permutation(self.DECK_CARDS)
        self._b_count: int = 0
        self._shuffle_idx = 0

    def step(self, Action) -> None:
        """
        Processes one agents turn and progresses the game to the next state.
        """

class Player:
    """
    Generate a default player.
    Player should have 106_000 made up of:
    - 1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 15000, 20000, 25000
    
    The player class will serve more as storage than an actual player.
    Actions will be handled by the engine with respect to the player.
    """
    def __init__(self) -> None:
        self._money = np.array([1,2,3,4,6,8,10,12,15,20,25])

    def has_money_card(self, val: int) -> bool:
        # Check if the player has the money card in their hand
        return len(np.where(self._money == val)[0]) > 0


if __name__ == "__main__":
    val = 16
    player = Player()
    print(player.has_money_card(val))