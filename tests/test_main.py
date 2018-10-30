from typing import List
import unittest
import sys

import main

ARGS: List[str] = ["main.py", "-o", "tmp"]


class TestMain(unittest.TestCase):
    def test_one_sr_exemplars(self) -> None:
        sys.argv = ARGS + ["-f", "tests/configurations/exemplars.yml"]
        main.main()

    def test_one_sr_expression(self) -> None:
        sys.argv = ARGS + ["-f", "tests/configurations/symbolic_regression.yml"]
        main.main()

    def test_one_sr_coev(self) -> None:
        sys.argv = ARGS + [
            "-f",
            "tests/configurations/coevolution_symbolic_regression.yml",
            "--coev",
        ]
        main.main()

    def test_one_iterated_prisoners_dilemma(self) -> None:
        sys.argv = ARGS + ["-f", "tests/configurations/iterated_prisoners_dilemma.yml"]
        main.main()

    def test_one_iterated_prisoners_dilemma_coev(self) -> None:
        sys.argv = ARGS + [
            "-f",
            "tests/configurations/coevolution_iterated_prisoners_dilemma.yml",
            "--coev",
        ]
        main.main()


if __name__ == "__main__":
    unittest.main()
