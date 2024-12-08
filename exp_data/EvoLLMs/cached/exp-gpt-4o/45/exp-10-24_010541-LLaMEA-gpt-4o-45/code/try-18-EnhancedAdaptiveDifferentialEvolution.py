import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 12 * dim  # Increased population size for better exploration
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.CR = np.random.uniform(0.6, 1.0, self.pop_size)  # Adjusted crossover rate range
        self.F = np.random.uniform(0.5, 0.8, self.pop_size)  # Adjusted scaling factor range
        self.local_intensification = 0.07  # Slightly increased local search probability

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                # Mutation with adaptive strategy
                indices = np.arange(self.pop_size)
                indices = indices[indices != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.population[a] + self.F[i] * (self.population[b] - self.population[c]) + np.random.normal(0, 0.1, self.dim)

                # Stochastic Local Search Intensification
                if np.random.rand() < self.local_intensification:
                    local_best_idx = np.random.choice(np.argsort(self.fitness)[:max(1, self.pop_size // 5)])
                    local_best = self.population[local_best_idx]
                    mutant = 0.5 * (mutant + local_best)

                mutant = np.clip(mutant, *self.bounds)

                # Dynamic Crossover
                j_rand = np.random.randint(self.dim)
                trial = np.where((np.random.rand(self.dim) < self.CR[i]) | (np.arange(self.dim) == j_rand), mutant, self.population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.CR[i] = 0.8 * self.CR[i] + 0.2 * np.random.rand()
                    self.F[i] = 0.8 * self.F[i] + 0.2 * np.random.rand()
                else:
                    self.CR[i] = 0.2 * self.CR[i] + 0.8 * np.random.rand()
                    self.F[i] = 0.2 * self.F[i] + 0.8 * np.random.rand()

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]