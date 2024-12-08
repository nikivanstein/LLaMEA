import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(4, 10 + 3 * int(np.log(dim)))  # Dynamic population size
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.CR = np.random.uniform(0.4, 0.9, self.pop_size)
        self.F = np.random.uniform(0.2, 0.8, self.pop_size)
        self.local_intensification = 0.07  # Slightly increased local search probability

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
                    mutant = 0.5 * (mutant + local_best)
                
                mutant = np.clip(mutant, *self.bounds)

                # Exponential Crossover
                crossover_start = np.random.randint(self.dim)
                crossover_length = 0
                while crossover_length < self.dim and (np.random.rand() < self.CR[i] or crossover_length == 0):
                    index = (crossover_start + crossover_length) % self.dim
                    trial = np.copy(self.population[i])
                    trial[index] = mutant[index]
                    crossover_length += 1

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