import json
import unittest

from fitness.program_synthesis.program_synthesis import FindCharacters, FindCharactersSymbolicExecution


class TestFindCharacters(unittest.TestCase):

    def test_run(self):
        data_file = 'tests/program_synthesis/FindCharacters.json'
        with open(data_file, 'r') as f:
            data = json.load(f)

        program_synthesis = FindCharacters(data['train'])
        code = """
    res0 = 0
    for i in inputs:
        if i == "a" or i == "b":
            res0 = res0 + 1
    return res0
    """
        result = program_synthesis.run(code)
        print(result)
        self.assertTrue(sum(result) == 100, result)


class TestFindCharactersSymbolicExecution(unittest.TestCase):

    def test_run(self):
        data_file = 'tests/program_synthesis/FindCharacters.json'
        with open(data_file, 'r') as f:
            data = json.load(f)

        program_synthesis = FindCharactersSymbolicExecution(data['train'], "")
        code = """
        self.increment = 2
        print(self.increment)
        res0 = 0
        for i in inputs:
            if i == "a" or i == "b":
                res0 = res0 + self.increment
        return res0
    """
        result = program_synthesis.run(code)
        print(result)
        self.assertTrue(sum(result) == 0, result)

        code = """
        self.increment = 1
        res0 = 0
        for i in inputs:
            if i == "a" or i == "b":
                res0 = res0 + self.increment
        return res0
    """
        result = program_synthesis.run(code)
        print(result)
        self.assertTrue(sum(result) == 100, result)
