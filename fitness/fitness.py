from fitness.symbolic_regression import SymbolicRegression

DEFAULT_FITNESS = -float("inf")


def mean(values):
    return sum(values) / len(values)


class SRFitness(object):
    def __init__(self, param):
        self.exemplars = eval(param["exemplars"])
        self.symbolic_expression = eval(param["symbolic_expression"])

    def __call__(self, fcn_str, cache):
        raise NotImplementedError("Define in subclass")

    def run(self, cache):
        key = "{}-{}".format(self.symbolic_expression, self.exemplars)
        if key in cache:
            fitness = cache[key]
        else:
            targets = [_[-1] for _ in self.exemplars]
            sr = SymbolicRegression(self.exemplars, self.symbolic_expression)
            predictions = sr.run()
            try:
                fitness = SRFitness.get_fitness(targets, predictions)
            except FloatingPointError:
                fitness = DEFAULT_FITNESS

            cache[key] = fitness

        return fitness

    @staticmethod
    def get_fitness(targets, predictions):
        errors = []
        for target, prediction in zip(targets, predictions):
            errors.append((target - prediction) ** 2)

        fitness = mean(errors)
        return fitness


class SRExpression(SRFitness):
    def __call__(self, fcn_str, cache):
        self.symbolic_expression = eval(fcn_str)
        fitness = self.run(cache)
        return fitness

    def coev(self, fcn_str, tests, cache):
        fitnesses = [DEFAULT_FITNESS] * len(tests)
        for i, test in enumerate(tests):
            self.exemplars = eval(test.phenotype)
            fitness = self.__call__(fcn_str, cache)
            fitnesses[i] = fitness

        fitness = mean(fitnesses)
        return fitness


class SRExemplar(SRFitness):
    def __call__(self, fcn_str, cache):
        self.exemplars = eval(fcn_str)
        fitness = self.run(cache)
        fitness = -fitness
        return fitness

    def coev(self, fcn_str, tests, cache):
        fitnesses = [DEFAULT_FITNESS] * len(tests)
        for i, test in enumerate(tests):
            self.symbolic_expression = eval(test.phenotype)
            fitness = self.__call__(fcn_str, cache)
            fitnesses[i] = fitness

        fitness = mean(fitnesses)
        return fitness


def get_fitness_function(param):
    """Returns fitness function object.

    Used to construct fitness functions from the configuration parameters

    :param param: Fitness function parameters
    :type param: dict
    :return: Fitness function
    :rtype: Object
    """
    name = param["name"]
    if name == "SRExpression":
        fitness_function = SRExpression(param)
    elif name == "SRExemplar":
        fitness_function = SRExemplar(param)
    else:
        raise ("Unknown fitness function: {}".format(name))

    return fitness_function


if __name__ == "__main__":
    pass
