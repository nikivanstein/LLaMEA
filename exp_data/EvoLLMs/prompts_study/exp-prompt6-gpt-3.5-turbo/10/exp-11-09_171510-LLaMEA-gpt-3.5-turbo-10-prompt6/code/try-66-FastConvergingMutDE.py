import numpy as np

class FastConvergingMutDE(EnhancedDynamicMutDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_range = np.random.uniform(0.1, 0.9, (self.population_size, dim))
        self.crossover_range = np.random.uniform(0.1, 0.9, (self.population_size, dim))

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.population_size):
                a, b, c = np.random.choice(self.population_size, 3, replace=False)
                mutant = self.population[a] + self.mutation_range[i] * (self.population[b] - self.population[c])
                trial = np.clip(mutant, -5.0, 5.0)

                crossover_points = np.random.rand(self.dim) < self.crossover_range[i]
                trial = np.where(crossover_points, trial, self.population[i])

                if func(trial) < func(self.population[i]):
                    self.population[i] = trial
                    self.mutation_range[i] += 0.2 if self.mutation_range[i] < 0.9 else -0.2
                    self.crossover_range[i] += 0.1 if self.crossover_range[i] < 0.9 else -0.1

        final_fitness = [func(individual) for individual in self.population]
        best_idx = np.argmin(final_fitness)
        best_individual = self.population[best_idx]

        return best_individual