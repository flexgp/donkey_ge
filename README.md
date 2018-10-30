# donkey_ge

Simple implementation of Grammatical Evolution. Uses `python3`. 

## Install

Install requirements
```
pip install -r requirements.txt
```

## Run

Paths are relative the repository root.

### Tutorials

See `tutorials`. The `tests` can also help understand the program.

### One way evolutionary search

Find a symbolic regression, i.e. a symbolic expression that matches the input and output.


#### Symbolic Expression
```
python main.py -f tests/configurations/symbolic_regression.yml -o results
```

#### Exemplars
```
python main.py -f tests/configurations/exemplars.yml -o results
```

### Two way evolutionary search - Coevolutionary

#### Symbolic Regression

The adversaries are coupled, i.e. dependent on each other.
```
python main.py -f tests/configurations/coevolution_symbolic_regression.yml -o results --coev
```

### `donkey_ge` output

`donkey_ge` prints some information to `stdout` regarding `settings` and
search progress for each iteration, see `donkey_ge.py:print_stats`. 

The output files have each generation as a list element, and each individual separated by a `,`. They are written to:
```
donkey_ge_*_fitness_values.json
donkey_ge_*_length_values.json
donkey_ge_*_size_values.json
donkey_ge_*_solution_values.json
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

Use `pre-commit.sh` as a pre-commit hook. E.g. `ln -s ../../pre-commit.sh .git/hooks/pre-commit`

## Documentation

See [docs/README.md](docs/README.md) for more details and basic
Evolutionary Computation background.
