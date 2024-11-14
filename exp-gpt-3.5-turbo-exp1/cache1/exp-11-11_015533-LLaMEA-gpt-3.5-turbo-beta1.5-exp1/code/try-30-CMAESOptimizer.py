import numpy as np
import cma

class CMAESOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        es = cma.CMAEvolutionStrategy(np.random.rand(self.dim), 0.5)
        while not es.stop() and es.countiter < self.budget:
            solutions = es.ask()
            fitness_values = [func(x) for x in solutions]
            es.tell(solutions, fitness_values)
        best_solution = es.best.get()[0]
        return best_solution