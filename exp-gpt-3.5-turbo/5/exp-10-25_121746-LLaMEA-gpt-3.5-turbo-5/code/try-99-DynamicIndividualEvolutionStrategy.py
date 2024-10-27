import numpy as np

class DynamicIndividualEvolutionStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.diversity_threshold = 0.05
        self.individual_evolution_rate = 0.05
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness = np.zeros(self.population_size)
        self.global_best = None

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.population_size):
                new_individual = self.population[i] + np.random.normal(0, self.individual_evolution_rate, self.dim)
                new_individual = np.clip(new_individual, -5.0, 5.0)
                new_fitness = func(new_individual)
                if new_fitness < self.fitness[i]:
                    self.population[i] = new_individual
                    self.fitness[i] = new_fitness
                    if self.global_best is None or new_fitness < func(self.global_best):
                        self.global_best = new_individual
        return self.global_best