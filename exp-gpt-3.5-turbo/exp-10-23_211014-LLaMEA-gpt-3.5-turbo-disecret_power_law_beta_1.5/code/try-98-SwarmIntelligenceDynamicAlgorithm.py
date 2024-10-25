import numpy as np

class SwarmIntelligenceDynamicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_iter = budget // self.population_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
    
    def __call__(self, func):
        for _ in range(self.max_iter):
            fitness = [func(individual) for individual in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]
            new_population = [individual + 0.1 * np.random.randn(self.dim) for individual in self.population]
            self.population = np.clip(new_population, self.lower_bound, self.upper_bound)
        return best_individual