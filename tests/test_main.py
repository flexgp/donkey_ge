from typing import List
import unittest
import sys

import main

ARGS: List[str] = ["-o", "tmp"]


class TestMain(unittest.TestCase):
    def test_one_sr_exemplars(self) -> None:
        _args = ARGS + ["-f", "tests/configurations/exemplars.yml"]
        main.main(_args)

    def test_one_sr_expression(self) -> None:
        _args = ARGS + ["-f", "tests/configurations/symbolic_regression.yml"]
        main.main(_args)

    def test_one_sr_coev(self) -> None:
        _args = ARGS + [
            "-f",
            "tests/configurations/coevolution_symbolic_regression.yml",
            "--coev",
        ]
        main.main(_args)

    def test_one_iterated_prisoners_dilemma(self) -> None:
        _args = ARGS + ["-f", "tests/configurations/iterated_prisoners_dilemma.yml"]
        main.main(_args)

    def test_one_iterated_prisoners_dilemma_coev(self) -> None:
        _args = ARGS + [
            "-f",
            "tests/configurations/coevolution_iterated_prisoners_dilemma.yml",
            "--coev",
        ]
        main.main(_args)

    def test_one_hawk_and_dove(self) -> None:
        _args = ARGS + ["-f", "tests/configurations/hawk_and_dove.yml"]
        main.main(_args)

    def test_one_hawk_and_dove_coev(self) -> None:
        _args = ARGS + [
            "-f",
            "tests/configurations/coevolution_hawk_and_dove.yml",
            "--coev",
        ]
        main.main(_args)

    def test_one_program_synthesis(self) -> None:
        _args = ARGS + ["-f", "tests/configurations/program_synthesis.yml"]
        main.main(_args)


if __name__ == "__main__":
    unittest.main()
