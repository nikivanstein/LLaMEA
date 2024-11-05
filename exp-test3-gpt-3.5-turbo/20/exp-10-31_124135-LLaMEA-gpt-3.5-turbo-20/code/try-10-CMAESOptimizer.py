import numpy as np
import cma

class CMAESOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        es = cma.CMAEvolutionStrategy(np.zeros(self.dim), 0.5, {'bounds': [-5, 5]})
        while not es.stop():
            solutions = es.ask()
            fitness_values = [func(x) for x in solutions]
            es.tell(solutions, fitness_values)
        return es.result.xbest