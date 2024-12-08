import numpy as np

class ImprovedDynamicMutDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))
        self.adaptive_factors = np.random.uniform(0.0, 1.0, (self.population_size, 2))
        self.threshold = 0.5

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.population_size):
                a, b, c = np.random.choice(self.population_size, 3, replace=False)
                mutant = self.population[a] + np.clip(self.adaptive_factors[i, 0], 0.1, 0.9) * (self.population[b] - self.population[c])
                trial = np.clip(mutant, -5.0, 5.0)

                crossover_points = np.random.rand(self.dim) < np.clip(self.adaptive_factors[i, 1], 0.1, 0.9)
                trial = np.where(crossover_points, trial, self.population[i])

                if func(trial) < func(self.population[i]):
                    self.population[i] = trial
                    if func(trial) < func(self.population[i]) - self.threshold:
                        self.adaptive_factors[i] += np.array([0.3, 0.2]) if np.all(self.adaptive_factors[i] < 1.0) else np.array([-0.3, -0.2])

        final_fitness = [func(individual) for individual in self.population]
        best_idx = np.argmin(final_fitness)
        best_individual = self.population[best_idx]

        return best_individual