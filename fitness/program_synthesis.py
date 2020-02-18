from abc import ABCMeta, abstractmethod


class ProgramSynthesis(object, metaclass=ABCMeta):

    def __init__(self, data):
        self.input = data['input']
        self.output = data['output']
        assert len(self.input) == len(self.output)

    @abstractmethod
    def run(self, source_code):
        pass


class FindCharacters(ProgramSynthesis):

    def run(self, source_code):
        pass