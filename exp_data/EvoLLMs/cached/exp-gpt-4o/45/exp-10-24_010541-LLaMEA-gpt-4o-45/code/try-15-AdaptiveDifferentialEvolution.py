# Description: Adaptive Differential Evolution with Enhanced Diversity Maintenance and Stochastic Recombination.
# Code:
import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.CR = np.random.uniform(0.5, 1.0, self.pop_size)
        self.F = np.random.uniform(0.4, 0.9, self.pop_size)
        self.local_intensification = 0.1  # Enhanced local search probability

    def __call__(self, func):
        evaluations = 0
        archive = []  # Diversity maintenance via archive

        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                # Mutation with enhanced diversity
                indices = np.arange(self.pop_size)
                indices = indices[indices != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.population[a] + self.F[i] * (self.population[b] - self.population[c])
                
                # Local Search Intensification (with archive influence)
                if np.random.rand() < self.local_intensification:
                    if archive:
                        archive_best = archive[np.argmin([func(ind) for ind in archive])]
                        mutant = 0.5 * (mutant + archive_best)
                
                mutant = np.clip(mutant, *self.bounds)

                # Stochastic Crossover
                j_rand = np.random.randint(self.dim)
                trial = np.where((np.random.rand(self.dim) < self.CR[i]) | (np.arange(self.dim) == j_rand), mutant, self.population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < self.fitness[i]:
                    # Update archive for diversity
                    archive.append(self.population[i])
                    if len(archive) > self.pop_size:
                        archive.pop(np.random.randint(len(archive)))

                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.CR[i] = 0.8 * self.CR[i] + 0.2 * np.random.rand()  # Adjustments for better stochastic control
                    self.F[i] = 0.8 * self.F[i] + 0.2 * np.random.rand()
                else:
                    self.CR[i] = 0.2 * self.CR[i] + 0.8 * np.random.rand()
                    self.F[i] = 0.2 * self.F[i] + 0.8 * np.random.rand()

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]