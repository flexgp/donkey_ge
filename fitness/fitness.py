"""Module for the fitness functions. A fitness function serves as an interface between
the engagement environment and the search heuristic. It provides the engagement
environment with the actions. It computes a fitness score based on the measurements
from the engagement environment.
"""
import json
from typing import List, Dict, Any, Tuple, Callable

from fitness.symbolic_regression import SymbolicRegression
from fitness.game_theory_game import PrisonersDilemma, HawkAndDove
from fitness.program_synthesis.program_synthesis import FindCharacters
from heuristics.donkey_ge import Individual, DEFAULT_FITNESS, FitnessFunction


def mean(values: List[float]) -> float:
    """
    Return the mean of the values.
    """
    return sum(values) / len(values)


class SRFitness(FitnessFunction):
    """
    Symbolic Regression fitness function.

    Attributes:
        exemplars: Exemplars
        symbolic_expression: Symbolic expression
    """

    def __init__(self, param: Dict[str, Any]) -> None:
        """
        Set class attributes for exemplars and symbolic expression
        """
        self.exemplars: List[List[float]] = eval(param["exemplars"])  # pylint: disable=eval-used
        self.symbolic_expression = eval(param["symbolic_expression"])  # pylint: disable=eval-used

    def __call__(self, fcn_str: str, cache: Dict[str, float]) -> float:
        raise NotImplementedError("Define in subclass")

    def run(self, cache: Dict[str, float]) -> float:
        """
        Evaluate exemplars with the symbolic expression.
        """
        key = "{}-{}".format(self.symbolic_expression, self.exemplars)
        if key in cache:
            fitness = cache[key]
        else:
            targets = [_[-1] for _ in self.exemplars]
            symbolic_regression = SymbolicRegression(self.exemplars, self.symbolic_expression)
            predictions = symbolic_regression.run()
            try:
                fitness = SRFitness.get_fitness(targets, predictions)
            except FloatingPointError:
                fitness = DEFAULT_FITNESS

            cache[key] = fitness

        return fitness

    @staticmethod
    def get_fitness(targets: List[float], predictions: List[float]) -> float:
        """
        Return mean squared error.
        """
        errors = []
        for target, prediction in zip(targets, predictions):
            errors.append((target - prediction) ** 2)

        fitness = mean(errors)
        return fitness


class SRExpression(SRFitness):
    """
    Symbolic expression
    """

    def __call__(self, fcn_str: str, cache: Dict[str, float]) -> float:
        """
        Evaluate symbolic expression and return negated fitness.
        """

        self.symbolic_expression = eval(fcn_str)  # pylint: disable=eval-used
        fitness = self.run(cache)
        fitness = -fitness
        return fitness

    def coev(self, fcn_str: str, tests: List[Individual], cache: Dict[str, float]) -> float:
        """
        Evaluate symbolic expression on all tests and return mean fitness.
        """
        fitnesses = [DEFAULT_FITNESS] * len(tests)
        for i, test in enumerate(tests):
            self.exemplars = eval(test.phenotype)  # pylint: disable=eval-used
            fitness = self.__call__(fcn_str, cache)
            fitnesses[i] = fitness

        fitness = mean(fitnesses)
        return fitness


class SRExemplar(SRFitness):
    """
    Exemplars, e.g. x[0], x[1], x[2], x[-1] is considered the target value
    """

    def __call__(self, fcn_str: str, cache: Dict[str, float]) -> float:
        """
        Evaluate exemplars and return fitness.
        """
        self.exemplars = eval(fcn_str)  # pylint: disable=eval-used
        fitness = self.run(cache)
        return fitness

    def coev(self, fcn_str: str, tests: List[Individual], cache: Dict[str, float]) -> float:
        """
        Evaluate exemplars on all tests and return mean fitness.
        """

        fitnesses = [DEFAULT_FITNESS] * len(tests)
        for i, test in enumerate(tests):
            self.symbolic_expression = eval(test.phenotype)  # pylint: disable=eval-used
            fitness = self.__call__(fcn_str, cache)
            fitnesses[i] = fitness

        fitness = mean(fitnesses)
        return fitness


class IteratedPrisonersDilemma(FitnessFunction):
    """
    Iterated Prisoners Dilemma fitness function

    Note: There are faster ways of evaluating, this is for demonstration purposes.

    Attributes:
        n_iterations: Number of iterations of the Prisoners Dilemma
        prisoners_dilemma: Prisoners Dilemma
        opponent: Strategy of opponent
        player: Strategy of player
    """

    def __init__(self, param: Dict[str, Any]) -> None:
        """ Initialize object
        """
        self.n_iterations = param["n_iterations"]
        self.prisoners_dilemma = PrisonersDilemma(self.n_iterations)
        self.opponent = eval(param["opponent"])  # pylint: disable=eval-used
        self.player = lambda x, y: ""

    def __call__(self, fcn_str: str, cache: Dict[str, float]) -> float:
        """ Evaluate the strategy against the opponent and return fitness.
        """
        key: str = "{}-{}".format(fcn_str, self.opponent)
        if key in cache:
            fitness: float = cache[key]
        else:
            self.player: Callable[
                [List[Tuple[str, str]], int], str
            ] = eval(  # pylint: disable=eval-used
                fcn_str
            )
            sentences, _ = self.prisoners_dilemma.run(self.player, self.opponent)
            fitness = IteratedPrisonersDilemma.get_fitness(sentences)
            cache[key] = fitness

        return fitness

    @staticmethod
    def get_fitness(sentences: List[Tuple[float, float]]) -> float:
        """ Fitness is the negated sum of the sentences
        """
        fitness: float = -sum([_[0] for _ in sentences])
        return fitness

    def coev(self, fcn_str: str, strategies: List[Individual], cache: Dict[str, float]) -> float:
        """ Evaluate one strategy against multiple strategies and mean expected utility (fitness).
        """
        fitnesses: List[float] = [DEFAULT_FITNESS] * len(strategies)
        for i, strategy in enumerate(strategies):
            self.opponent: Callable[
                [List[Tuple[str, str]], int], str
            ] = eval(  # pylint: disable=eval-used
                strategy.phenotype
            )
            fitnesses[i] = self.__call__(fcn_str, cache)

        # Mean Expected Utility
        fitness: float = mean(fitnesses)
        return fitness


class IteratedHawkAndDove(FitnessFunction):
    """
    Hawk and Dove fitness function

    Note: There are faster ways of evaluating, this is for demonstration purposes.

    Attributes:
        n_iterations: Number of iterations of the game
        hawk_and_dove: Hawk And Dove game
        opponent: Strategy of opponent
        player: Strategy of player
    """

    def __init__(self, param: Dict[str, Any]) -> None:
        """ Initialize object
        """
        self.n_iterations = param["n_iterations"]
        self.hawk_and_dove = HawkAndDove(self.n_iterations)
        self.opponent = eval(param["opponent"])  # pylint: disable=eval-used
        self.player = lambda x, y: ""

    def __call__(self, fcn_str: str, cache: Dict[str, float]) -> float:
        """ Evaluate the strategy against the opponent and return fitness.
        """
        key: str = "{}-{}".format(fcn_str, self.opponent)
        if key in cache:
            fitness: float = cache[key]
        else:
            self.player: Callable[
                [List[Tuple[str, str]], int], str
            ] = eval(  # pylint: disable=eval-used
                fcn_str
            )
            payoff, _ = self.hawk_and_dove.run(self.player, self.opponent)
            fitness = IteratedHawkAndDove.get_fitness(payoff)
            cache[key] = fitness

        return fitness

    @staticmethod
    def get_fitness(payoffs: List[Tuple[float, float]]) -> float:
        """ Fitness is the negated sum of the payoff
        """
        fitness: float = -sum([_[0] for _ in payoffs])
        return fitness

    def coev(self, fcn_str: str, strategies: List[Individual], cache: Dict[str, float]) -> float:
        """ Evaluate one strategy against multiple strategies and mean expected utility (fitness).
        """
        fitnesses: List[float] = [DEFAULT_FITNESS] * len(strategies)
        for i, strategy in enumerate(strategies):
            self.opponent: Callable[
                [List[Tuple[str, str]], int], str
            ] = eval(  # pylint: disable=eval-used
                strategy.phenotype
            )
            fitnesses[i] = self.__call__(fcn_str, cache)

        # Mean Expected Utility
        fitness: float = mean(fitnesses)
        return fitness

class ProgramSynthesis(FitnessFunction):
    """
    Program Synthesis fitness function.
    """

    def __init__(self, param: Dict[str, Any]) -> None:
        """
        Initialize object
        """
        self.data = param['data']
        if isinstance(self.data, str) and self.data.endswith('.json'):
            with open(self.data, 'r') as f:
                self.data = json.load(f)

        self.data_split = self.get_input_and_output_split('train')
        self.code_template_path = param.get('code_template_path', '')
        if self.code_template_path:
            with open(self.code_template_path, 'r') as f:
                self.code_template = f.readlines()
                self.code_template = ''.join(self.code_template)
        else:
            self.code_template = ''
        self.program_synthesis = FindCharacters(self.data_split, self.code_template)

    def get_input_and_output_split(self, split: str) -> Dict[str, Any]:
        return self.data[split]

    def __call__(self, fcn_str: str, cache: Dict[str, float]) -> float:
        """
        Evaluate program based on input and output exemplars
        """
        key: str = "{}".format(fcn_str)
        if key in cache:
            fitness: float = cache[key]
        else:
            result = self.program_synthesis.run(fcn_str)
            fitness = ProgramSynthesis.get_fitness(result)
            cache[key] = fitness

        return fitness

    @staticmethod
    def get_fitness(outcomes: List[bool]) -> float:
        """ Fitness is the sum of correct exemplars
        """
        fitness: float = sum(outcomes)
        return fitness


if __name__ == "__main__":
    pass
