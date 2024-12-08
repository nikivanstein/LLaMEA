import numpy as np

class EnhancedDEAdaptiveImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.f = 0.5
        self.cr = 0.9
        self.population_size = 10
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))  # Pre-allocate population array
        fitness_values = np.empty(self.budget)  # Pre-allocate fitness_values array

        for i in range(self.budget):
            fitness_values[i] = func(population[i])

        for _ in range(self.population_size, self.budget):
            mutants_idx = np.random.choice(self.population_size, (self.population_size, 2), replace=True)
            crossover = np.random.rand(self.population_size, self.dim) < self.cr
            mutants = population[mutants_idx]

            new_population = population[:self.population_size] + self.f * (mutants[:, 0] - mutants[:, 1])
            new_population = np.where(crossover, np.clip(new_population, self.lower_bound, self.upper_bound), population[:self.population_size])

            new_fitness_values = np.empty(self.population_size)
            for i in range(self.population_size):
                new_fitness_values[i] = func(new_population[i])

            improved_indices = new_fitness_values < fitness_values[:self.population_size]
            population[:self.population_size][improved_indices] = new_population[improved_indices]
            fitness_values[:self.population_size][improved_indices] = new_fitness_values[improved_indices]

        best_index = np.argmin(fitness_values)
        return population[best_index]