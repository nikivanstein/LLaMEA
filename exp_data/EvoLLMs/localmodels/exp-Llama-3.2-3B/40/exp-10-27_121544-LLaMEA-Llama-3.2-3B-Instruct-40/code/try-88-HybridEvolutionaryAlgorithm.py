import numpy as np
from scipy.optimize import differential_evolution

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.population = [self.x0]
        self.adaptive_strategy = np.random.uniform(0.0, 1.0, size=self.dim)

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        new_individuals = []
        for individual in self.population:
            new_individuals.append(individual)
            if np.random.rand() < 0.4:
                new_individual = individual + np.random.uniform(-1.0, 1.0, size=self.dim)
                new_individuals.append(new_individual)

        self.population = new_individuals
        self.adaptive_strategy = np.mean(self.population, axis=0)

        min_value = np.inf
        best_individual = None
        for individual in self.population:
            value = func(individual)
            if value < min_value:
                min_value = value
                best_individual = individual

        return min_value, best_individual

# Example usage:
budget = 100
dim = 10
func = lambda x: x[0]**2 + x[1]**2
heacombbo = HybridEvolutionaryAlgorithm(budget, dim)
min_value, best_individual = heacombbo(func)
print(f"Minimum value: {min_value}, Best individual: {best_individual}")