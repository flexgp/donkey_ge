import json
from typing import List
import os

from matplotlib import pyplot as plt
import numpy as np


def plot_fitness(
    out_path: str = ".",
    in_path: str = ".",
) -> None:
    """
    Plot the fitness per generation
    """
    files: List[str] = os.listdir(in_path)

    for _file in files:
        # TODO: fix path to file, which is not the same for a coev fitness value
        if _file == "donkey_ge_fitness_values.json":
            file_path: str = os.path.join(in_path, _file)
            with open(file_path, 'r') as in_file:
                data = json.load(in_file)

            fitness = np.array(data["fitness_values"])
            plt.subplot(1, 1, 1)
            plt.title("Fitness per generation")
            # Best fitness
            ys = fitness[:, 0]
            plt.plot(ys, label="Best fitness")
            # Average fitness per generation
            ys = np.mean(fitness, 1)
            plt.plot(ys, label="Mean fitness")
            plt.xlabel("Generation")
            plt.ylabel("Fitness")
            plt.legend()
            plot_name = _file.replace(".json", ".pdf")
            plt.savefig(os.path.join(out_path, plot_name))


if __name__ == '__main__':
    plot_fitness()
