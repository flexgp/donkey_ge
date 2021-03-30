
import argparse
import contextlib
import json
import os
import sys
from io import StringIO
import random
from typing import List, Dict, Any
import signal

from z3 import z3

class ParenthesisMatching():
    """
    Use symbolic execution to find characters
    """
    
    TIMEOUT = 100

    def __init__(self, data: Dict[str, List[Any]], code_template) -> None:
        self.solver = z3.Solver()
        self.inputs = data["inputs"]
        self.outputs = data["output"]
        assert len(self.inputs) == len(self.outputs)
        self.code_template = """
class Cls:

    def __init__(self):
        self.increment = 0
        self.decrement = 0
        self.check = 0

    def fcn(self, inputs):
        {}

    def run(self, inputs, outputs):
        self.outcomes = evaluate_exemplars(inputs, outputs, self)
        return self.outcomes

instance = Cls()
outcomes = instance.run(inputs, outputs)
"""
        if code_template:
            self.code_template = code_template

    @contextlib.contextmanager
    def stdoutIO(self, stdout=None) -> None:
        """ From https://stackoverflow.com/questions/3906232/python-get-the-print-output-in-an-exec-statement"""
        old = sys.stdout
        if stdout is None:
            stdout = StringIO()
        sys.stdout = stdout
        yield stdout
        sys.stdout = old

    def run_handler(self, signal_number: int, frame: Any) -> RuntimeError:
        raise RuntimeError(f"Error for run_handler signal: {signal_number}")

    def run(self, source_code: str) -> Dict[str, Any]:
        result = {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "evaluate_exemplars": self.evaluate_exemplars,
        }
        print("aewugwaygfu")
        program = self.code_template.format(source_code)
        if __debug__:
            print(f"Run program:\n{program}")
        with self.stdoutIO() as sio:
            try:
                signal.signal(signal.SIGALRM, self.run_handler)
                signal.alarm(ParenthesisMatching.TIMEOUT)
                exec(program, result)  # pylint: disable=exec-used
            except RuntimeError as e:
                print(f"TimeoutError {e} for:\n{program}")
                result["outcomes"] = None

        e_so = sio.getvalue()
        if e_so:
            print(f"From exec:\n{e_so}")

        return result["outcomes"]

    def evaluate_exemplars(self, inputs, outputs, instance):
        outcomes = []
        solver = None
        for _input, _output in zip(inputs, outputs):
            outcome = instance.fcn(_input[0])
            # TODO rewrite so solver can be called before needing to evaluate
            if solver is None:
                solver = z3.Solver()
                solver.add(instance.increment == instance.decrement)
                _c = solver.check()
                if _c == z3.sat:
                    print(f"Solver model:{solver.model()}")
                else:
                    break

            outcomes.append(outcome == _output[0])

        return outcomes

    @staticmethod
    def parenthesis_matching(s: str) -> int:
        """
        TODO: Update!
        ```
        """
        left = 0
        for i in s:
            if i == "(":
                left += 1
            elif i == ")":
                if left == 0: # every before has been fully matched (but we see another right parenthesis)
                    return False
                else:
                    left -= 1
        return left == 0 # fully matched

    
    @classmethod
    def main(cls, n_generate: int, out_path: str) -> None:
        MAX_CNT = 100_000
        assert 0 < n_generate < MAX_CNT
        problem_set = cls.__name__
        N_m = 3
        N = 20
        data_path = os.path.join(out_path, f"{problem_set}.json")
        data = {"train": None, "test": None}
        for data_split in data.keys():
            exemplars = {"inputs": [], "output": []}
            data[data_split] = exemplars
            cnt = 0
            while len(exemplars["inputs"]) < n_generate and cnt < MAX_CNT:
                cnt += 1
                n = random.randint(N_m, N)
                _input = "".join(random.choices(["(", ")"], k=n))
                _output = ParenthesisMatching.parenthesis_matching(_input)
                exemplars["inputs"].append([_input])
                exemplars["output"].append([_output])

            if len(exemplars["inputs"]) < n_generate:
                raise Exception(f"Too few exemplars {len(exemplars['inputs'])} < {n_generate}")

        with open(data_path, "w") as fd:
            json.dump(data, fd)


class ParenthesisMatchingNoSolver(ParenthesisMatching):
    def evaluate_exemplars(self, inputs, outputs, instance):
        outcomes = []
        for _input, _output in zip(inputs, outputs):
            outcome = instance.fcn(_input[0])
            outcomes.append(outcome == _output[0])

        return outcomes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate solutions and inputs for find characters"
    )
    parser.add_argument(
        "--n_generate", type=int, required=True, help="Number of Test and Train samples generated"
    )
    parser.add_argument("--out_path", type=str, required=True, help="Path to output files e.g .")

    args = parser.parse_args()

    ParenthesisMatching.main(args.n_generate, args.out_path)