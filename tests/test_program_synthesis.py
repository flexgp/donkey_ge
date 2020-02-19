import json
import unittest

from fitness.program_synthesis.program_synthesis import FindCharacters


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
