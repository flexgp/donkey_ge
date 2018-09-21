from typing import List, Callable


class SymbolicRegression:
    def __init__(
        self, exemplars: List[List[float]], symbolic_expression: Callable[[List[float]], float]
    ) -> None:
        self.exemplars = exemplars
        self.symbolic_expression = symbolic_expression

    def run(self) -> List[float]:
        outputs = []  # List[float]
        for exemplar in self.exemplars:
            output = self.symbolic_expression(exemplar)
            outputs.append(output)

        return outputs
