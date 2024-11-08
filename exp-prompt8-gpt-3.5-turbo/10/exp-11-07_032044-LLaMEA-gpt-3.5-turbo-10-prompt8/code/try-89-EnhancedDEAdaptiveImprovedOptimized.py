import numpy as np

class EnhancedDEAdaptiveImprovedOptimized:
    def __init__(self, budget, dim):
        self.budget, self.dim, self.f, self.cr = budget, dim, 0.5, 0.9
        self.population_size, self.lower_bound, self.upper_bound = 10, -5.0, 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness_values = np.array([func(ind) for ind in population])

        for _ in range(self.budget - self.population_size):
            mutants_idx = np.random.randint(self.population_size, size=(self.population_size, 2))
            crossover = np.random.rand(self.population_size, self.dim) < self.cr
            mutants = population[mutants_idx]

            new_population = population + self.f * (mutants[:, 0] - mutants[:, 1])
            np.clip(new_population, self.lower_bound, self.upper_bound, out=new_population)
            np.copyto(new_population, population, where=np.logical_not(crossover))

            new_fitness_values = np.array([func(ind) for ind in new_population])
            improved_indices = new_fitness_values < fitness_values
            np.place(population, improved_indices, new_population[improved_indices])
            np.place(fitness_values, improved_indices, new_fitness_values[improved_indices])

        best_index = np.argmin(fitness_values)
        return population[best_index]