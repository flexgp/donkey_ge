from typing import List, Dict, Any

from fitness.symbolic_regression import SymbolicRegression
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


if __name__ == "__main__":
    pass
