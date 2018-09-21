import collections
import time
import random
import os
from typing import Any, List, Dict
from numbers import Number

from heuristics.donkey_ge import (
    map_input_with_grammar,
    sort_population,
    print_stats,
    tournament_selection,
    Individual,
    variation,
    generational_replacement,
    Grammar,
    initialise_population,
    parse_arguments,
    Population,
)
from fitness import fitness

__author__ = "Erik Hemberg"
"""
Alternating Coevolutionary Algorithm
"""

# TODO better way of setting cache max size
CACHE_MAX_SIZE = 100000


class Population(object):
    """A population container"""

    def __init__(
        self,
        fitness_function: Any,
        grammar: Grammar,
        adversary: str,
        name: str,
        individuals: List[Individual],
    ) -> None:
        """Container for a population.
        :param fitness_function:
        :type fitness_function: function
        :param grammar:
        :type grammar: Grammar
        :param adversary:
        :type adversary: str
        :param name:
        :type name: str
        :param individuals:
        :type individuals: list of Individual
        """
        self.fitness_function = fitness_function
        self.grammar = grammar
        self.adversary = adversary
        self.name = name
        self.individuals = individuals

    def clone(self) -> Population:
        clone = Population(
            self.fitness_function, self.grammar, self.adversary, self.name, self.individuals
        )
        return clone

    def __str__(self) -> str:
        individuals = "\n".join(map(str, self.individuals))
        _str = "{} {} {} {}\n{}".format(
            str(self.fitness_function),
            self.grammar.file_name,
            self.adversary,
            self.name,
            individuals,
        )

        return _str


def evaluate(
    individual: Individual,
    fitness_function: Any,
    inds: List[Individual] = [],
    cache: Dict[str, float] = None,
) -> Individual:
    """Evaluates phenotype in fitness_function function and sets fitness_function.
    :param individual:
    :type individual: Individual
    :param fitness_function: Fitness function
    :type fitness_function: function
    :param inds: Other individuals
    :type inds: list of Individuals
    :param cache: Cache for evaluation speed-up
    :type cache: dict
    :return: individual
    :rtype: Individual
    """

    individual.fitness = fitness_function.coev(individual.phenotype, inds, cache)

    assert individual.fitness is not None

    return individual


def evaluate_fitness(
    individuals: List[Individual],
    grammar: Grammar,
    fitness_function: Any,
    adversaries: List[Individual] = [],
    param: Dict[str, Any] = {},
):
    """Perform the fitness evaluation for each individual of the population.
    :param individuals:
    :type individuals: list of Individual
    :param grammar:
    :type grammar: Grammar
    :param fitness_function:
    :type fitness_function: function
    :param adversaries: Competitors (or collaborators) of individual
    :type adversaries: list of Individuals
    :param param: Other parameters
    :type param: dict
    :return: Evaluated indviduals
    :rtype: list of Individuals
    """

    # TODO efficient caching for parallel evaluation
    cache = param["cache"]

    n_individuals = len(individuals)
    # Iterate over all the individual solutions
    for ind in individuals:
        # TODO map only once
        map_input_with_grammar(ind, grammar)
        assert ind.phenotype
        if ind.phenotype is not "":
            # Execute the fitness function
            evaluate(ind, fitness_function, adversaries, cache)
            assert ind.fitness is not None

    assert n_individuals == len(individuals), "%d != %d" % (n_individuals, len(individuals))

    return individuals


def search_loop_coevolution(
    populations: Dict[str, Population], param: Dict[str, Any]
) -> Dict[str, Individual]:
    """Return the best individual from the evolutionary search
    loop.
    :param populations: Initial populations for search
    :type populations: dict of str and Population
    :param param: Parameters for search
    :type param: dict
    :return: Best individuals
    :rtype: dict
    """

    # Evaluate fitness
    param["cache"] = collections.OrderedDict()

    stats_dict: collections.OrderedDict[str, Any] = collections.OrderedDict()
    best_ever: collections.OrderedDict[str, Individual] = collections.OrderedDict()

    for key, population in populations.items():
        start_time = time.time()
        stats_dict[key] = collections.defaultdict(list)
        stats = stats_dict[key]
        grammar = population.grammar
        fitness_function = population.fitness_function
        adversary = populations[population.adversary]
        for ind in adversary.individuals:
            map_input_with_grammar(ind, adversary.grammar)

        population.individuals = evaluate_fitness(
            population.individuals, grammar, fitness_function, adversary.individuals, param
        )
        # Set best solution
        population.individuals = sort_population(population.individuals)
        best_ever[key] = population.individuals[0]

        # Print the stats of the populations
        print(key, len(param["cache"]))
        print_stats(0, population.individuals, stats, start_time)

    # Generation loop
    generation = 1
    while generation < param["generations"]:
        if len(param["cache"]) > CACHE_MAX_SIZE:
            param["cache"].clear()

        for key, population in populations.items():
            start_time = time.time()
            stats = stats_dict[key]
            grammar = population.grammar
            fitness_function = population.fitness_function
            adversary = populations[population.adversary]
            for ind in adversary.individuals:
                map_input_with_grammar(ind, adversary.grammar)

            # Selection
            parents = tournament_selection(
                population.individuals, param["population_size"], param["tournament_size"]
            )

            elites = [Individual(_.genome) for _ in population.individuals[: param["elite_size"]]]

            # TODO do not bother with elite_number of variations
            new_individuals = variation(parents, param)

            for i in range(len(elites)):
                new_individuals[i] = elites[i]

            # Evaluate fitness
            new_individuals = evaluate_fitness(
                new_individuals, grammar, fitness_function, adversary.individuals, param
            )

            # Replace populations

            # Fitness is relative the adversaries, thus an elite must
            # always be re-evaluated
            population.individuals = generational_replacement(
                new_individuals,
                population.individuals,
                population_size=param["population_size"],
                elite_size=0,
            )

            # Set best solution
            population.individuals = sort_population(population.individuals)
            best_ever[key] = population.individuals[0]

            # Print the stats of the populations
            print(key, len(param["cache"]))
            print_stats(generation, population.individuals, stats, start_time)

        # Increase the generation counter
        generation += 1

    write_run_output(generation, stats_dict, populations, param)

    best_solution_str = ["%s: %s" % (k, v) for k, v in best_ever.items()]
    print("Best solution: %s" % (",".join(best_solution_str)))

    return best_ever


def write_run_output(
    generation: int,
    stats_dict: Dict[str, Dict[str, List[Number]]],
    populations: Dict[str, Population],
    param: Dict[str, Any],
) -> None:
    """Write run stats to files.
    :param generation: Generation number
    :type generation: int
    :param stats_dict: Collected statistics of run
    :type stats_dict: dict
    :param populations: Populations
    :type populations: dict of str and Population
    :param param: Parameters
    :type param: dict
    """
    _hist: Dict[str, int] = collections.defaultdict(int)
    for k, v in param["cache"].items():
        _hist[str(v)] += 1

    print(
        "Cache entries:%d Total Fitness Evaluations:%d Fitness Values:%d"
        % (
            len(param["cache"].keys()),
            generation * param["population_size"] ** 2,
            len(_hist.keys()),
        )
    )

    out_file_name = "donkey_ge"
    if "output_dir" in param:
        output_dir = param["output_dir"]
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        out_file_name = os.path.join(output_dir, out_file_name)

    _out_file_name = "%s_settings.out" % out_file_name
    with open(_out_file_name, "w") as out_file:
        for k, v in param.items():
            if k != "cache":
                out_file.write("%s: %s\n" % (k, str(v)))

    for key in populations.keys():
        stats = stats_dict[key]
        for k, v in stats.items():
            _out_file_name = "%s_%s_%s.csv" % (out_file_name, key, k)
            with open(_out_file_name, "w") as out_file:
                for line in v:
                    if k == "solution_values":
                        out_file.write("%s\n" % (";".join(map(str, line))))
                    else:
                        out_file.write("%s\n" % (",".join(map(str, line))))


def run(param: Dict[str, Any]) -> Dict[str, Individual]:
    """
    Return the best solution. Create an initial
    population. Perform an evolutionary search.

    :param param: parameters for pony gp
    :type param: dict
    :returns: Best solution
    :rtype: dict
    """

    start_time = time.time()

    # Set random seed if not 0 is passed in as the seed
    if "seed" not in param.keys():
        param["seed"] = int(time.time())

    random.seed(param["seed"])
    print("Setting random seed: %d %.5f" % (param["seed"], random.random()))

    # Print settings
    print("donkey_ge settings:", param)

    assert param["population_size"] > 1
    assert param["generations"] > 0
    assert param["max_length"] > 0
    assert param["seed"] > -1
    assert param["integer_input_element_max"] > 0
    assert param["tournament_size"] <= param["population_size"]
    assert param["elite_size"] < param["population_size"]
    assert 0.0 <= param["crossover_probability"] <= 1.0
    assert 0.0 <= param["mutation_probability"] <= 1.0

    ###########################
    # Create initial population
    ###########################
    populations: collections.OrderedDict = collections.OrderedDict()
    for key, value in param["populations"].items():
        p_dict = param["populations"][key]
        grammar = Grammar(p_dict["bnf_grammar"])
        grammar.read_bnf_file(grammar.file_name)
        fitness_function = fitness.get_fitness_function(p_dict["fitness_function"])
        adversary = p_dict["adversary"]
        Individual.max_length = param["max_length"]
        Individual.codon_size = param["integer_input_element_max"]
        individuals = initialise_population(param["population_size"])
        population = Population(fitness_function, grammar, adversary, key, individuals)
        populations[key] = population

    ###########################
    # Evolutionary search
    ###########################
    best_ever = search_loop_coevolution(populations, param)

    # Display results
    print("Time: %.3f Best solution:%s" % (time.time() - start_time, best_ever))

    return best_ever


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
