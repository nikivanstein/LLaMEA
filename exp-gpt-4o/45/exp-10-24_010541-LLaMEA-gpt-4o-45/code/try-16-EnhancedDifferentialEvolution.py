import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 12 * dim
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.CR = np.random.uniform(0.5, 1.0, self.pop_size)
        self.F = np.random.uniform(0.5, 0.8, self.pop_size)
        self.local_intensification = 0.1
        self.success_rate = np.zeros(self.pop_size)

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                # Mutation
                indices = np.arange(self.pop_size)
                indices = indices[indices != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.population[a] + self.F[i] * (self.population[b] - self.population[c])
                
                # Local Search Intensification
                if np.random.rand() < self.local_intensification:
                    local_best = self.population[np.argmin(self.fitness)]
                    mutant = np.mean([mutant, local_best], axis=0)
                
                mutant = np.clip(mutant, *self.bounds)

                # Crossover
                j_rand = np.random.randint(self.dim)
                trial = np.where((np.random.rand(self.dim) < self.CR[i]) | (np.arange(self.dim) == j_rand), mutant, self.population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.CR[i] = 0.85 * self.CR[i] + 0.15 * np.random.rand()
                    self.F[i] = 0.85 * self.F[i] + 0.15 * np.random.rand()
                    self.success_rate[i] += 1
                else:
                    self.CR[i] = 0.15 * self.CR[i] + 0.85 * np.random.rand()
                    self.F[i] = 0.15 * self.F[i] + 0.85 * np.random.rand()

            # Adaptive control mechanism
            if evaluations % (self.pop_size * 2) == 0:
                mean_success_rate = np.mean(self.success_rate)
                self.local_intensification = max(0.05, self.local_intensification * (0.5 + mean_success_rate))
                self.success_rate.fill(0)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]