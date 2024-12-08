import numpy as np
from cma import CMAEvolutionStrategy

class DynamicCMALocalSearch:
    def __init__(self, budget, dim, population_size=30):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size

    def __call__(self, func):
        es = CMAEvolutionStrategy(0.5 * np.random.randn(self.dim), 0.1, {'popsize': self.population_size})
        while not es.stop():
            solutions = es.ask()
            fitness_values = [func(x) for x in solutions]
            es.tell(solutions, fitness_values)
        return es.result[0]