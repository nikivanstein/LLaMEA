import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.best_idx = None
        self.best_value = np.inf
        self.evals = 0

    def __call__(self, func):
        for i in range(self.pop_size):
            self.fitness[i] = func(self.population[i])
        self.evals += self.pop_size
        self.best_idx = np.argmin(self.fitness)
        self.best_value = self.fitness[self.best_idx]

        while self.evals < self.budget:
            for i in range(self.pop_size):
                if self.evals >= self.budget:
                    break

                # Mutation: select three random individuals
                idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                a, b, c = self.population[idxs]

                # Dynamic scaling of F
                self.F = 0.4 + 0.1 * np.random.rand()

                # Generate mutant vector
                mutant = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])

                # Selection
                trial_fitness = func(trial)
                self.evals += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                    # Update the best solution found
                    if trial_fitness < self.best_value:
                        self.best_idx = i
                        self.best_value = trial_fitness

        return self.population[self.best_idx]