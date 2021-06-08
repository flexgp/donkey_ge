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

class Dijkstra():
    """
    Synthesize shortest path that resembles Dijkstra's algorithm
    """
    
    TIMEOUT = 100

    def __init__(self, data: Dict[str, List[Any]], code_template) -> None:
        self.solver = z3.Solver()
        self.inputs = data["inputs"]
        self.outputs = data["output"]
        assert len(self.inputs) == len(self.outputs)
        self.code_template = """
def find_shortest_distance(adj, s):
        d = [0,100000, 100000, 10000]
        counter = 0
        for i in ([{}]):
            counter += 1
            updated = False
            for j in ([0,1,2,3]):
                if d[i] + adj[i][j] < d[j]:
                    d[j] = d[i] + adj[i][j]
                    updated = True
            if not updated:
                return d[s], counter
        return d[s], counter

outcomes = evaluate_exemplars(inputs, outputs, find_shortest_distance)
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
                signal.alarm(Dijkstra.TIMEOUT)
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
            outcome, count = instance(_input[0], _input[1])
            if outcome != _output[0]:
                outcomes.append(0)
            else:
                outcomes.append(min((_output[1]/count), 1))
                # outcomes.append(1)
        return outcomes
        
    @staticmethod
    def find_shortest_distance(adj, s):
        d = [0,100000, 100000, 10000]
        counter = 0
        for i in ([0,1,2,3]):
            counter += 1
            updated = False 
            for j in ([0,1,2,3]):
                if d[i] + adj[i][j] < d[j]:
                    d[j] = d[i] + adj[i][j]
                    updated = True
            if not updated:
                return d[s], counter
        return d[s], counter

    
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
                weight = 0
                for i in range(4):
                    adj.append([])
                    for  j in range(4):
                        increment = random.randint(-1,1)
                        if increment == -1:
                            adj[-1].append(10000)
                        else:
                            weight += increment
                            adj[-1].append(weight)
                print(adj)
                s = random.randint(1,3)
                _output = Dijkstra.find_shortest_distance(adj, s)
                exemplars["inputs"].append([adj, s])
                exemplars["output"].append(_output)

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

    Dijkstra.main(100, "tests/program_synthesis/Dijkstra_Efficient.json")