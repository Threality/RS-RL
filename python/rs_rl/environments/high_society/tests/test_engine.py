import unittest

import numpy as np

from rs_rl.environments.high_society.actions import BidAction, PassAction
from rs_rl.environments.high_society.engine import Environment, Player


class PlayerTests(unittest.TestCase):
    def test_bid_removes_and_counts_money_card(self) -> None:
        player = Player()

        player.play_money_card(10)

        self.assertFalse(player.has_money_card(10))
        self.assertEqual(player.spent, 10)

    def test_remove_card_modifier_removes_highest_prestige_card(self) -> None:
        player = Player()
        player.add_card("3")
        player.add_card("8")

        player.add_card("b_rem_card")

        self.assertEqual(player.cards, ("3",))


class EngineTests(unittest.TestCase):
    def make_environment(self, cards: list[str]) -> Environment:
        env = Environment(seed=123)
        env._deck = np.array(cards, dtype=np.str_)
        env._shuffle_idx = 0
        env._auction_active = False
        return env

    def test_normal_auction_bid_is_paid_and_card_is_awarded(self) -> None:
        env = self.make_environment(["1", "2"])

        env.step(BidAction(4))
        env.step(PassAction())
        env.step(PassAction())
        env.step(PassAction())

        self.assertEqual(env._players[0].spent, 4)
        self.assertEqual(env._players[0].cards, ("1",))
        self.assertEqual(env._shuffle_idx, 1)
        self.assertFalse(env.auction_active)

    def test_bad_card_is_taken_by_passer_for_free(self) -> None:
        env = self.make_environment(["b_neg_5"])

        env.step(PassAction())

        self.assertTrue(env.terminated)
        self.assertEqual(env._players[0].cards, ("b_neg_5",))
        self.assertEqual(env._players[0].spent, 0)

    def test_reset_is_reproducible(self) -> None:
        first = Environment(seed=987)
        second = Environment(seed=987)

        self.assertEqual(first._deck.tolist(), second._deck.tolist())

    def test_invalid_bid_does_not_mutate_state(self) -> None:
        env = self.make_environment(["1"])

        with self.assertRaises(ValueError):
            env.step(BidAction(99))

        self.assertFalse(env.auction_active)
        self.assertEqual(env._b_count, 0)
        self.assertEqual(env._players[0].spent, 0)


if __name__ == "__main__":
    unittest.main()
