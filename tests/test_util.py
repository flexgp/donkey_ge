from typing import List, Tuple
import os
import unittest

from util.plot_ipd import plot_iterated_prisoners_dilemma, plot_ipd_from_file
from fitness.prisoners_dilemma import PrisonersDilemma


class TestUtil(unittest.TestCase):
    def test_one_plot_iterated_prisoners_dilemma(self) -> None:
        C = PrisonersDilemma.COOPERATE
        D = PrisonersDilemma.DEFECT
        sentences: List[Tuple[float, float]] = [(1.0, 1.0), (3.0, 0.0), (0.0, 3.0), (2.0, 2.0)]
        histories: List[Tuple[str, str]] = [(C, C), (C, D), (D, C), (C, C)]
        out_path: str = "."
        expected_file_name: str = "ipd_test.pdf"
        if os.path.exists(os.path.join(out_path, expected_file_name)):
            os.remove(expected_file_name)
        plot_iterated_prisoners_dilemma(sentences, histories, out_path)
        self.assertTrue(os.path.exists(expected_file_name))

    def test_one_plot_ipd_from_file(self) -> None:
        file_name = "test_ipd.json"
        player_1 = lambda h, i: "C"
        player_2 = lambda h, i: "C" if h[i][0] == "C" else "D"
        n_iterations = 2
        memory_size = 1
        expected_file_name = "test_one_ipd.pdf"
        runs = 2
        pd = PrisonersDilemma(
            n_iterations=n_iterations,
            memory_size=memory_size,
            store_stats=True,
            out_file_name=file_name,
        )
        for _ in range(runs):
            _, _ = pd.run(player_1=player_1, player_2=player_2)
        plot_ipd_from_file(file_name, out_path=".", name=expected_file_name)
        for i in range(runs):
            self.assertTrue(os.path.exists("{}_{}".format(i, expected_file_name)))


if __name__ == "__main__":
    unittest.main()
