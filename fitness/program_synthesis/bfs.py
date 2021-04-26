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

class BFS():
    """
    Synthesize a graph exploration program that resembles BFS
    """
    
    TIMEOUT = 100

    def __init__(self, data: Dict[str, List[Any]], code_template) -> None:
        self.solver = z3.Solver()
        self.inputs = data["inputs"]
        self.outputs = data["output"]
        assert len(self.inputs) == len(self.outputs)
        self.code_template = """
def find_all_neighbors(adj, s):
    neighbors = set()
    for i in ([{}]):
        if adj[s][i]:
            neighbors.add(i)
    output = list(neighbors)
    output.sort()
    return output
outcomes = evaluate_exemplars(inputs, outputs, find_all_neighbors)
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
        program = self.code_template.format(source_code)
        if __debug__:
            print(f"Run program:\n{program}")
        with self.stdoutIO() as sio:
            try:
                signal.signal(signal.SIGALRM, self.run_handler)
                signal.alarm(BFS.TIMEOUT)
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
        for _input, _output in zip(inputs, outputs):
            outcome = instance(_input[0], _input[1])
            score = 0
            for i in outcome:
                if i in _output[0]:
                    score += 1
            if  len(_output[0]) == 0 and score == 0:
                    outcomes.append(1)
            elif len(outcome) != len(_output[0]):
                outcomes.append(0)
            else:
                outcomes.append(score / len(_output[0]))
        return outcomes
        
    @staticmethod
    def find_all_neighbors(adj, s):
        neighbors = set()
        for i in ([0,1,2,3]):
            if adj[s][i]:
                neighbors.add(i)
        output = list(neighbors)
        output.sort()
        return output

    @staticmethod
    def find_one_neighbor(adj, s):
        for i in ([0,1,2,3]):
            if adj[s][i]:
                return [i]
        return []

    
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
                adj = []
                for i in range(4):
                    adj.append([])
                    for  j in range(4):
                        adj[-1].append(random.randint(0,1))
                s = random.randint(0,3)
                adj[s][0] = 0
                _output = BFS.find_all_neighbors(adj, s)
                exemplars["inputs"].append([adj, s])
                exemplars["output"].append([_output])

            if len(exemplars["inputs"]) < n_generate:
                raise Exception(f"Too few exemplars {len(exemplars['inputs'])} < {n_generate}")

        with open(out_path, "w") as fd:
            json.dump(data, fd)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Generate solutions and inputs for find characters"
    # )
    # parser.add_argument(
    #     "--n_generate", type=int, required=True, help="Number of Test and Train samples generated"
    # )
    # parser.add_argument("--out_path", type=str, required=True, help="Path to output files e.g .")

    # args = parser.parse_args()

    BFS.main(100, "tests/program_synthesis/all_neighbors_biased.json")