import numpy as np

class StochasticAdaptiveSearchStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        evaluations = 0
        while evaluations < self.budget:
            fitness_values = [func(individual) for individual in population]
            best_idx = np.argmin(fitness_values)
            best_individual = population[best_idx]
            centroid = np.mean(population, axis=0)
            population = 0.9 * population + 0.1 * centroid + 0.1 * np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
            evaluations += self.budget
        return best_individual