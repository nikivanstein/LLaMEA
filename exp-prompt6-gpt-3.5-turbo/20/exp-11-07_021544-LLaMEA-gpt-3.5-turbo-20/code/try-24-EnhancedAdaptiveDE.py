import numpy as np

class EnhancedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.f = 0.5
        self.cr = 0.9

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget // self.population_size):
            idxs = np.random.randint(self.population_size, size=(self.population_size, 3))
            selected_population = population[idxs]
            mutants = np.clip(selected_population[:, 0] + self.f * (selected_population[:, 1] - selected_population[:, 2]), -5.0, 5.0)
            crossovers = np.random.rand(self.population_size, self.dim) < self.cr
            population = np.where(crossovers, mutants, population)
            fitness = np.where(crossovers, np.array([func(individual) for individual in population]), fitness)

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        return best_solution, best_fitness