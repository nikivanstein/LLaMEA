import numpy as np

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.CR = np.random.uniform(0.4, 1.0, self.pop_size)
        self.F = np.random.uniform(0.5, 0.9, self.pop_size)
        self.neighborhood_radius = 0.2  # probability of neighborhood exploration
        self.dynamic_learning = 0.15

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                # Mutation with dynamic learning
                indices = np.arange(self.pop_size)
                indices = indices[indices != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                learn_factor = 1.0 + self.dynamic_learning * (np.random.rand() - 0.5)
                mutant = self.population[a] + learn_factor * self.F[i] * (self.population[b] - self.population[c])
                
                # Adaptive Neighborhood Exploration
                if np.random.rand() < self.neighborhood_radius:
                    neighborhood = np.random.choice(indices, 5, replace=False)
                    neighborhood_best = self.population[neighborhood[np.argmin(self.fitness[neighborhood])]]
                    mutant = 0.5 * (mutant + neighborhood_best)
                
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
                    self.CR[i] = 0.9 * self.CR[i] + 0.1 * np.random.rand()  # enhanced adaptation
                    self.F[i] = 0.9 * self.F[i] + 0.1 * np.random.rand()
                else:
                    self.CR[i] = 0.3 * self.CR[i] + 0.7 * np.random.rand()
                    self.F[i] = 0.3 * self.F[i] + 0.7 * np.random.rand()

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]