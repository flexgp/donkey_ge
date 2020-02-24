import importlib


def import_function(fitness_function_str: str) -> str:
    module, method = fitness_function_str.rsplit(".", 1)
    fitness_function = importlib.import_module(module)
    method = getattr(fitness_function, method)
    return method
