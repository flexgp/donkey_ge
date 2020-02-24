import argparse
import contextlib
import json
import os
import string
import sys
from io import StringIO
import random
from typing import List, Dict, Any
import signal

from z3 import z3


class FindCharacters(object):

    TIMEOUT = 100

    def __init__(self, data: Dict[str, List[Any]], code_template) -> None:
        self.inputs = data['inputs']
        self.outputs = data['output']
        assert len(self.inputs) == len(self.outputs)
        self.code_template = """
    def fcn(inputs):
        {}

    outcomes = evaluate_exemplars(inputs, outputs, fcn)
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
        result = {'inputs': self.inputs, 'outputs': self.outputs,
                  'evaluate_exemplars': self.evaluate_exemplars}
        program = self.code_template.format(source_code)
        print(program)
        with self.stdoutIO() as sio:
            try:
                signal.signal(signal.SIGALRM, self.run_handler)
                signal.alarm(FindCharacters.TIMEOUT)
                exec(program, result)  # pylint: disable=exec-used
            except RuntimeError as e:
                print(f'TimeoutError {e} for:\n{program}')
                result['outcomes'] = None

        e_so = sio.getvalue()
        if e_so:
            print(f'From exec:\n{e_so}')

        return result['outcomes']

    def evaluate_exemplars(self, inputs, outputs, fcn):
        outcomes = []
        for _input, _output in zip(inputs, outputs):
            outcome = fcn(_input[0])
            outcomes.append(outcome == _output[0])

        return outcomes

    @staticmethod
    def find_characters(s: str) -> int:
        """
        Assume `s` is a string of lower case characters.

        Write a program that prints the number of times `'a'` and `'b'` occurs in `s`. For example, if `s = 'azcb'`,
        then your program should print
        ```
        Number of 'a' and 'b': 2
        ```
        """
        ctr = 0
        for i in s:
            if i == "a" or i == "b":
                ctr = ctr + 1
        print("Number of vowels:", ctr)
        return ctr

    @classmethod
    def main(cls, n_generate: int, out_path: str) -> None:
        MAX_CNT = 100000
        assert 0 < n_generate < MAX_CNT
        problem_set = cls.__name__
        N_m = 3
        N = 20
        data_path = os.path.join(out_path, f'{problem_set}.json')
        data = {'train': None, 'test': None}
        for data_split in data.keys():
            exemplars = {'inputs': [], 'output': []}
            data[data_split] = exemplars
            cnt = 0
            while len(exemplars['inputs']) < n_generate and cnt < MAX_CNT:
                cnt += 1
                n = random.randint(N_m, N)
                _input = ''.join(random.choices(string.ascii_lowercase, k=n))
                _output = FindCharacters.find_characters(_input)
                exemplars['inputs'].append([_input])
                exemplars['output'].append([_output])

            if len(exemplars['inputs']) < n_generate:
                raise Exception(f"Too few exemplars {len(exemplars['inputs'])} < {n_generate}")

        with open(data_path, 'w') as fd:
            json.dump(data, fd)


class FindCharactersSymbolicExecution(FindCharacters):
    """
    Use symbolic execution to find characters
    """
    def __init__(self, data: Dict[str, List[Any]], code_template) -> None:
        super(FindCharactersSymbolicExecution, self).__init__(data, code_template)
        self.solver = z3.Solver()
        self.code_template = """
class Cls:

    def __init__(self):
        self.increment = 0
        
    def fcn(self, inputs):
        {}
        
    def run(self, inputs, outputs):
        self.outcomes = evaluate_exemplars(inputs, outputs, self)
        print(self.increment)
        return self.outcomes
        
instance = Cls()
outcomes = instance.run(inputs, outputs)
"""
        if code_template:
            self.code_template = code_template

    def evaluate_exemplars(self, inputs, outputs, instance):
        outcomes = []
        for _input, _output in zip(inputs, outputs):
            outcome = instance.fcn(_input[0])
            solver = z3.Solver()
            solver.add(instance.increment == 1)
            _c = solver.check()
            if _c == z3.sat:
                print(solver.model())
            else:
                return outcomes

            outcomes.append(outcome == _output[0])

        return outcomes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate solutions and inputs for find characters')
    parser.add_argument('--n_generate', type=int, required=True,
                        help='Number of Test and Train samples generated')
    parser.add_argument('--out_path', type=str, required=True,
                        help='Path to output files e.g .')

    args = parser.parse_args()

    FindCharacters.main(args.n_generate, args.out_path)
