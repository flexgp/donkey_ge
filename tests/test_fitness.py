import unittest

from fitness.fitness import ProgramSynthesis
from fitness.program_synthesis.program_synthesis import FindCharactersSymbolicExecution


class TestProgramSynthesis(unittest.TestCase):

    def test_call(self):
        data_file = 'tests/program_synthesis/FindCharacters.json'
        param = {'data': data_file,
                 'code_template_path': 'tests/program_synthesis/code_template.txt',
                 'synthesis_problem': 'fitness.program_synthesis.program_synthesis.FindCharacters'}
        program_synthesis = ProgramSynthesis(param)
        fcn_str = """
        if i == "a" or i == "b":
        """
        fitness = program_synthesis(fcn_str, {})
        print(fitness)
        self.assertTrue(100, fitness)


class TestProgramSynthesisSymbolicExecution(unittest.TestCase):

    def test_call(self):
        data_file = 'tests/program_synthesis/FindCharacters.json'
        param = {'data': data_file,
                 'code_template_path': 'tests/program_synthesis/code_template_symbolic_execution.txt',
'synthesis_problem': 'fitness.program_synthesis.program_synthesis.FindCharactersSymbolicExecution'}
        program_synthesis = ProgramSynthesis(param)
        fcn_str = """2"""
        fitness = program_synthesis(fcn_str, {})
        print(fitness)
        self.assertTrue(100, fitness)
