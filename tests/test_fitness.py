import json
import unittest

from fitness.fitness import ProgramSynthesis


class TestProgramSynthesis(unittest.TestCase):

    def test_call(self):
        data_file = 'tests/program_synthesis/FindCharacters.json'
        param = {'data': data_file}
        program_synthesis = ProgramSynthesis(param)
        fcn_str = """
    res0 = 0
    for i in inputs:
        if i == "a" or i == "b":
            res0 = res0 + 1
    return res0
    """
        fitness = program_synthesis(fcn_str, {})
        print(fitness)
        self.assertTrue(100, fitness)
