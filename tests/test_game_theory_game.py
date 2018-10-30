import os
import unittest

from fitness.game_theory_game import PrisonersDilemma, HawkAndDove


class TestPrisonersDilemma(unittest.TestCase):
    def test_one_prisoners_dilemma(self) -> None:
        player_1 = lambda h, i: "C"
        player_2 = lambda h, i: "C" if h[i] == "C" else "D"
        n_iterations = 2
        memory_size = 1
        expected_sentences = [
            (PrisonersDilemma.S, PrisonersDilemma.T),
            (PrisonersDilemma.R, PrisonersDilemma.R),
        ]
        expected_history = {
            "player_1": ["","C", "C"],
            "player_2": ["", "D", "C"]
        }
        pd = PrisonersDilemma(
            n_iterations=n_iterations,
            memory_size=memory_size,
            store_stats=True,
            out_file_name=PrisonersDilemma.DEFAULT_OUT_FILE,
        )
        sentences, history = pd.run(player_1=player_1, player_2=player_2)
        for s, es in zip(sentences, expected_sentences):
            self.assertEqual(s, es)

        for h, eh in zip(history, expected_history):
            self.assertEqual(h, eh)

        self.assertTrue(os.path.exists(PrisonersDilemma.DEFAULT_OUT_FILE))


class TestHawkAndDove(unittest.TestCase):
    def test_one_hawk_and_dove(self) -> None:
        player_1 = lambda h, i: "H"
        player_2 = lambda h, i: "H" if h[i] == "H" else "D"
        n_iterations = 2
        memory_size = 1
        _payoff = HawkAndDove.PAYOFF
        expected_payoffs = [
            _payoff[(HawkAndDove.HAWK, HawkAndDove.DOVE)],
            _payoff[(HawkAndDove.HAWK, HawkAndDove.HAWK)],
        ]
        expected_history = {
            "player_1": ["","H", "H"],
            "player_2": ["", "D", "H"]
        }
        pd = HawkAndDove(
            n_iterations=n_iterations,
            memory_size=memory_size,
            store_stats=True,
            out_file_name=HawkAndDove.DEFAULT_OUT_FILE,
        )
        payoffs, history = pd.run(player_1=player_1, player_2=player_2)
        for s, es in zip(payoffs, expected_payoffs):
            self.assertEqual(s, es)

        for h, eh in zip(history, expected_history):
            self.assertEqual(h, eh)

        self.assertTrue(os.path.exists(HawkAndDove.DEFAULT_OUT_FILE))


if __name__ == "__main__":
    unittest.main()
