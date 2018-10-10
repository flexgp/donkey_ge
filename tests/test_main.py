import unittest
import main
import sys


ARGS = ["main.py", "-o", "tmp"]


class TestMain(unittest.TestCase):
    def test_one_sr_exemplars(self):
        sys.argv = ARGS + ["-f", "tests/configurations/exemplars.yml"]
        main.main()

    def test_one_sr_expression(self):
        sys.argv = ARGS + \
                   ["-f", "tests/configurations/symbolic_regression.yml"]
        main.main()

    def test_one_sr_coev(self):
        sys.argv = ARGS + [
            "-f",
            "tests/configurations/coevolution_symbolic_regression.yml",
            "--coev",
        ]
        main.main()


if __name__ == "__main__":
    unittest.main()
