class SymbolicRegression:
    def __init__(self, exemplars, symbolic_expression):
        self.exemplars = exemplars
        self.symbolic_expression = symbolic_expression

    def run(self):
        outputs = []
        for exemplar in self.exemplars:
            output = self.symbolic_expression(exemplar)
            outputs.append(output)

        return outputs
