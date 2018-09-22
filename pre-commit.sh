black --line-length 100 --py36 main.py fitness/ heuristics/ tests/test_main.py tests/test_hypothesis_donkey_ge.py

flake8 --max-line-length=120 main.py fitness/ heuristics/

mypy --strict .
