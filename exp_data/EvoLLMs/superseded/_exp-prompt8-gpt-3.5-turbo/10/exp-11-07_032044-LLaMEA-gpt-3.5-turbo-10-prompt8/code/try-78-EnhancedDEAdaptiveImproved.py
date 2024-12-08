import numpy as np

class EnhancedDEAdaptiveImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        f = 0.5
        cr = 0.9
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness_values = np.array([func(ind) for ind in population])

        for _ in range(self.budget - self.population_size):
            mutants_idx = np.random.choice(self.population_size, (self.population_size, 2), replace=True)
            crossover = np.random.rand(self.population_size, self.dim) < cr
            mutants = population[mutants_idx]

            new_population = population + f * (mutants[:, 0] - mutants[:, 1])
            new_population = np.where(crossover, np.clip(new_population, self.lower_bound, self.upper_bound), population)

            new_fitness_values = np.array([func(ind) for ind in new_population])
            improved_indices = new_fitness_values < fitness_values
            population[improved_indices] = new_population[improved_indices]
            fitness_values[improved_indices] = new_fitness_values[improved_indices]

            # Dynamic adaptation of mutation and crossover rates
            successful_rate = np.sum(improved_indices) / self.population_size
            f = 0.1 + 0.9 * successful_rate
            cr = 0.1 + 0.8 * successful_rate

        best_index = np.argmin(fitness_values)
        return population[best_index]