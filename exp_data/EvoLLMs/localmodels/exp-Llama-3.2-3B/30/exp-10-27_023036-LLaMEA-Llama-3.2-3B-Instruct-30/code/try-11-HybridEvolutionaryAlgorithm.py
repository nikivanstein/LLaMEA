import numpy as np
from scipy.optimize import differential_evolution

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.cma_es = CMAES(self.dim)
        self.es = EvolutionStrategies(self.dim)

    def __call__(self, func):
        population_size = 100
        bounds = [(-5.0, 5.0)] * self.dim
        n_calls = self.budget
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        result = differential_evolution(func, bounds, x0=np.random.uniform(-5.0, 5.0, size=population_size), 
                                         method='SDE', n_calls=n_calls, constraints=constraints)
        best_solution = result.x
        best_score = result.fun
        return best_solution, best_score

class CMAES:
    def __init__(self, dim):
        self.dim = dim
        self.cma = CMAES()

    def __call__(self, func):
        self.cma.optimize(func, self.dim)

class EvolutionStrategies:
    def __init__(self, dim):
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=self.dim)

    def __call__(self, func):
        self.x = self.es.update(self.x, func)

    def update(self, x, func):
        self.x = self.es.update(x, func)
        return self.x

# Load BBOB test suite
from bbob.benchmark import benchmark

def bbbenchmark(func, dim, budget, maxeval):
    algorithm = HybridEvolutionaryAlgorithm(budget, dim)
    results = []
    for i in range(maxeval):
        result = algorithm(func)
        results.append(result)
    return results

# Run the benchmark
results = benchmark(bbbenchmark, 24, 100, 100)
print(results)