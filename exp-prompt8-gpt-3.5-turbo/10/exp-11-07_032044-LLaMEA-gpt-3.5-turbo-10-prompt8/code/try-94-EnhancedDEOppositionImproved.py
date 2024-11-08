import numpy as np

class EnhancedDEOppositionImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.f = 0.5
        self.cr = 0.9
        self.population_size = 10
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness_values = np.array([func(ind) for ind in population])

        for _ in range(self.budget - self.population_size):
            mutants_idx = np.random.choice(self.population_size, (self.population_size, 2), replace=True)
            crossover = np.random.rand(self.population_size, self.dim) < self.cr
            mutants = population[mutants_idx]

            new_population = population + self.f * (mutants[:, 0] - mutants[:, 1])
            new_population = np.where(crossover, np.clip(new_population, self.lower_bound, self.upper_bound), population)

            # Opposition-based learning
            opposite_population = self.lower_bound + self.upper_bound - population
            opposite_fitness_values = np.array([func(ind) for ind in opposite_population])
            better_opposite = opposite_fitness_values < fitness_values
            population[better_opposite] = opposite_population[better_opposite]
            fitness_values[better_opposite] = opposite_fitness_values[better_opposite]

            new_fitness_values = np.array([func(ind) for ind in new_population])
            improved_indices = new_fitness_values < fitness_values
            population[improved_indices] = new_population[improved_indices]
            fitness_values[improved_indices] = new_fitness_values[improved_indices]

        best_index = np.argmin(fitness_values)
        return population[best_index]