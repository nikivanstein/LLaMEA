import numpy as np

class VectorizedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        return best_solution