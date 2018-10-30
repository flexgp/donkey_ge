import os
import unittest

from fitness.prisoners_dilemma import PrisonersDilemma


class TestPrisonersDilemma(unittest.TestCase):
    def test_one_prisoners_dilemma(self) -> None:
        player_1 = lambda h, i: "C"
        player_2 = lambda h, i: "C" if h[i][0] == "C" else "D"
        n_iterations = 2
        memory_size = 1
        expected_sentences = [
            (PrisonersDilemma.S, PrisonersDilemma.T),
            (PrisonersDilemma.R, PrisonersDilemma.R),
        ]
        expected_history = [("", ""), ("C", "D"), ("C", "C")]
        pd = PrisonersDilemma(n_iterations=n_iterations, memory_size=memory_size, store_stats=True)
        sentences, history = pd.run(player_1=player_1, player_2=player_2)
        for s, es in zip(sentences, expected_sentences):
            self.assertEqual(s, es)

        for h, eh in zip(history, expected_history):
            self.assertEqual(h, eh)

        self.assertTrue(os.path.exists(PrisonersDilemma.DEFAULT_OUT_FILE))


if __name__ == "__main__":
    unittest.main()
