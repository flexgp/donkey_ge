# donkey_ge

Simple implementation of Grammatical Evolution. Uses `python3`. 

## Install

Install requirements
```
pip install -r requirements.txt
```

## Run

Find a symbolic regression, i.e. a symbolic expression that matches the input and output.

### One way evolutionary search

####Symbolic Expression
```
python main.py -f tests/configurations/symbolic_regression.yml -o results
```

Gives output like:
```
Setting random seed: 1 0.13436
donkey_ge settings: {'population_size': 4, 'max_length': 100, 'elite_size': 1, 'generations': 3, 'tournament_size': 2, 'seed': 1, 'crossover_probability': 0.8, 'mutation_probability': 0.1, 'codon_size': 127, 'integer_input_element_max': 1000, 'bnf_grammar': 'tests/grammars/symbolic_regression.bnf', 'fitness_function': {'name': 'SRExpression', 'exemplars': '[(x1, x2, x1**2 + x2**2) for x1, x2 in zip(range(-11, 1), range(0, 10))]', 'symbolic_expression': 'None'}, 'output_dir': 'results', 'coev': False}
Gen:0 t:0.000 fit_ave:6278.45+-459.250 size_ave:21.00+-30.643 length_ave:100.00+-0.000 Ind: lambda x: x[1] - 5; 6737.7
Gen:1 t:0.001 fit_ave:7740.08+-1736.164 size_ave:27.00+-38.717 length_ave:92.75+-32.897 Ind: lambda x: x[0] * 4 * 5 * 0 - 4 + x[0] + x[0] + 1 - 0 * 0 * 3 - 0 * 4 * 4 * 5 - x[1] - 2 + 4 - 0 - x[1] + 2 * x[0] * 0 * 5; 10747.2
Gen:2 t:0.001 fit_ave:7266.08+-2153.728 size_ave:37.00+-33.985 length_ave:130.25+-30.078 Ind: lambda x: x[0] * 4 * 5 * 0 - 4 + x[0] + x[0] + 1 - 0 * 0 * 3 - 0 * 4 * 4 * 5 - x[1] - 2 + 4 - 0 - x[1] + 2 * x[0] * 0 * 5; 10747.2
Cache entries:4 Total Fitness Evaluations:48 Fitness Values:4
Time: 0.006 Best solution:Ind: lambda x: x[0] * 4 * 5 * 0 - 4 + x[0] + x[0] + 1 - 0 * 0 * 3 - 0 * 4 * 4 * 5 - x[1] - 2 + 4 - 0 - x[1] + 2 * x[0] * 0 * 5; 10747.2
```

####Exemplars
```
python main.py -f tests/configurations/symbolic_regression.yml -o results
```

Gives output like:
```
Setting random seed: 1 0.13436
donkey_ge settings: {'population_size': 4, 'max_length': 10, 'elite_size': 1, 'generations': 3, 'tournament_size': 2, 'seed': 1, 'crossover_probability': 0.8, 'mutation_probability': 0.1, 'codon_size': 127, 'integer_input_element_max': 1000, 'bnf_grammar': 'tests/grammars/exemplars.bnf', 'fitness_function': {'name': 'SRExemplar', 'symbolic_expression': 'lambda x: x[0]**2 + x[1]**2', 'exemplars': 'None'}, 'output_dir': 'results', 'coev': False}
Gen:0 t:0.001 fit_ave:-3007.77+-1513.825 size_ave:2.50+-0.866 length_ave:10.00+-0.000 Ind: [(x1, x2, x1**2) for x1, x2 in zip(range(-11, 1), range(0, 10))]; -1533.3
Gen:1 t:0.001 fit_ave:-1533.30+-0.000 size_ave:2.00+-0.000 length_ave:10.25+-0.829 Ind: [(x1, x2, x1**2) for x1, x2 in zip(range(-11, 1), range(0, 10))]; -1533.3
Gen:2 t:0.001 fit_ave:-1533.30+-0.000 size_ave:2.00+-0.000 length_ave:11.00+-0.000 Ind: [(x1, x2, x1**2) for x1, x2 in zip(range(-11, 1), range(0, 10))]; -1533.3
Cache entries:4 Total Fitness Evaluations:48 Fitness Values:4
Time: 0.005 Best solution:Ind: [(x1, x2, x1**2) for x1, x2 in zip(range(-11, 1), range(0, 10))]; -1533.3
```

### Two way evolutionary search - Coevolutionary

The adversaries are coupled, i.e. dependent on each other.
```
python main.py -f tests/configurations/coevolution_symbolic_regression.yml -o results --coev
```

Gives output like:
```
Setting random seed: 1 0.13436
donkey_ge settings: {'population_size': 4, 'max_length': 100, 'generations': 2, 'tournament_size': 2, 'seed': 1, 'crossover_probability': 0.8, 'mutation_probability': 0.1, 'codon_size': 127, 'integer_input_element_max': 1000, 'elite_size': 1, 'populations': {'attacker': {'adversary': 'defender', 'bnf_grammar': 'tests/grammars/exemplars.bnf', 'fitness_function': {'name': 'SRExemplar', 'symbolic_expression': 'None', 'exemplars': 'None'}}, 'defender': {'adversary': 'attacker', 'bnf_grammar': 'tests/grammars/symbolic_regression.bnf', 'fitness_function': {'name': 'SRExpression', 'exemplars': 'None', 'symbolic_expression': 'None'}}}, 'output_dir': 'results', 'coev': True}
attacker 8
Gen:0 t:0.003 fit_ave:-1529.65+-1716.539 size_ave:3.00+-1.000 length_ave:100.00+-0.000 Ind: [(x1, x2, x1 - x1) for x1, x2 in zip(range(-11, 1), range(0, 10))]; -29.75
defender 10
Gen:0 t:0.002 fit_ave:1286.90+-0.000 size_ave:4.00+-3.464 length_ave:100.00+-0.000 Ind: lambda x: 3; 1286.9
attacker 12
Gen:1 t:0.003 fit_ave:-69.62+-69.066 size_ave:4.00+-0.000 length_ave:100.00+-0.000 Ind: [(x1, x2, x1 - x1) for x1, x2 in zip(range(-11, 1), range(0, 10))]; -29.75
defender 13
Gen:1 t:0.002 fit_ave:22.75+-0.000 size_ave:3.00+-1.732 length_ave:100.00+-0.000 Ind: lambda x: 3; 22.75
Cache entries:13 Total Fitness Evaluations:32 Fitness Values:8
Best solution: attacker: Ind: [(x1, x2, x1 - x1) for x1, x2 in zip(range(-11, 1), range(0, 10))]; -29.75,defender: Ind: lambda x: 3; 22.75
Time: 0.019 Best solution:OrderedDict([('attacker', <heuristics.donkey_ge.Individual object at 0x10b55bac8>), ('defender', <heuristics.donkey_ge.Individual object at 0x10b7a7fd0>)])
```

### `donkey_ge` output

`donkey_ge` prints some information to `stdout` regarding `settings` and
search progress for each iteration, see `donkey_ge.py:print_stats`. 

The output files have each generation on a row, and each individual separated by a `,`. They are written to:
```
donkey_ge_*_fitness_values.csv
donkey_ge_*_length_values.csv
donkey_ge_*_size_values.csv
donkey_ge_*_solution_values.csv
```

### Usage
```
python main.py -h
usage: main.py [-h] -f CONFIGURATION_FILE [-o OUTPUT_DIR] [--coev]

Run donkey_ge

optional arguments:
  -h, --help            show this help message and exit
  -f CONFIGURATION_FILE, --configuration_file CONFIGURATION_FILE
                        YAML configuration file. E.g.
                        configurations/demo_ge.yml
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Path to directory for output files. E.g.
                        donkey_ge_output
  --coev           Coevolution
  Use multiprocessing for fitness evaluation
```

### Settings

Configurations are in `.yml` format, see examples in folder [configurations](tests/configurations).

Grammar is in *Backus-Naur Form (BNF)*, see examples in folder [grammars](tests/grammars)

## Test

Tests are in `tests` folder. E.g. run with `pytest`
```
pytest tests
```

Some tests are written with *Hypothesis*, http://hypothesis.readthedocs.io/en/master/index.html

## Development

For formatting use `black` e.g.
```
black --line-length 100 --py36 main.py fitness/ heuristics/ tests/test_main.py tests/test_hypothesis_fitness.py tests/test_hypothesis_mule_ge.py
```

Use `flake8`, e.g.
```
flake8 main.py fitness/ heuristics/
```


## Documentation

See [docs/README.md](docs/README.md) for more details and basic
Evolutionary Computation background.
