import numpy as np

class EnhancedDEAdaptiveImprovedEfficientInit:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.f = 0.5
        self.cr = 0.9
        self.population_size = 10
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness_values = np.array([func(ind) for ind in self.population])

    def __call__(self, func):
        for _ in range(self.budget - self.population_size):
            mutants_idx = np.random.choice(self.population_size, (self.population_size, 2), replace=True)
            crossover = np.random.rand(self.population_size, self.dim) < self.cr
            mutants = self.population[mutants_idx]

            new_population = self.population + self.f * (mutants[:, 0] - mutants[:, 1])
            new_population = np.where(crossover, np.clip(new_population, self.lower_bound, self.upper_bound), self.population)

            new_fitness_values = np.array([func(ind) for ind in new_population])
            improved_indices = new_fitness_values < self.fitness_values
            self.population[improved_indices] = new_population[improved_indices]
            self.fitness_values[improved_indices] = new_fitness_values[improved_indices]

        best_index = np.argmin(self.fitness_values)
        return self.population[best_index]