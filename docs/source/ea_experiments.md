# Running experiments with `donkey_ge`

Description of how to run experiments with `donkey_ge`

## Stochastic Search Heuristics

A *heuristic* is a technique for solving a problem:

- quickly when other methods are too inefficient
- for finding an approximate solution when other methods fail to
  find any exact solution

The *heuristic* achieves this by relaxing the:

- optimality, an optimal solution is not guaranteed
- completeness, all the optimal solutions might not be found
- accuracy, what are the accuracy of the solutions
- execution time, how fast is a solution returned

## When to use *Grammatical Evolution*

Grammatical Evolution is a form of *Genetic Programming* that use a grammar to

- Non-linear optimization and modelling

  - No analytic method

- Guarantee syntactical correctness of individual solutions

### Key components

- Representation

- Fitness function

- Implementation

### Search space size

The search space of `donkey_ge` is all the possible combinations of tree
sizes and shapes up until max depth

## Design of a Donkey GE Experiment


# `donkey_ge` run

A run refers to executing the Genetic Program once.

- Quality of best solution

- Statistics from the population as the search progresses over
  generations

An example of output from `main.py` run shows:
- The settings used by `donkey_ge.py`
- the generation number `Gen`
- average fitness plus/minus the standard deviation of the population
  `fit_ave`
- the best individual's fitness and bitstring `Individual`

```
python main.py -f configurations/dev_network_attack.yml
TBD
```

The following output files are created:
```
head donkey_ge*csv
TBD
```

## `donkey_ge` experiment

- Create a BNF-form grammar file in `grammars` folder

- Create a configuration file in `yml` in `configurations` folder

  - Set parameters for the search heuristic

	- Number of fitness evaluations - magnitude of search

	  - `population_size` - how many solutions to evaluate in parallel

	  - `generations` - how many iterations the population will be
		modified

	- Number of variations - variation frequency,

	  - `mutation_probability` - amplitude and search bias is determined by the operator

	  - `crossover_probability` - amplitude and search bias is determined by the operator

	- Selection pressure - convergence speed of search

	  - Selection operator

		- `tournament_size` - how many solutions from the current population that are competing for each place in the next population

		- `elite_size` - how many solutions are preserved between generations

  - Representation - how a solution is represented. For GE we use

	- `max_length` - how long a solution can be

	- `integer_input_element_max` - the integer range of an element in the solution

	- `bnf_grammar` - the grammar used to map the solution to a string

  - Fitness function - method to evaluate performance

	- `fitness_function` - the function used to evaluate the performance of a solution

  - `seed` - Seed used for random number generator, usefull for replicating results

- Determine the stability of search parameters, the evolutionary search is
  stochastic. Perform a number of independent runs to gain confidence
  in the results
