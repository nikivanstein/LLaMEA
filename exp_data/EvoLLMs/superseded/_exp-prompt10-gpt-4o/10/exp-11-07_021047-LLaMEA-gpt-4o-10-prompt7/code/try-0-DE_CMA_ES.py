import numpy as np

class DE_CMA_ES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 4 + int(3 * np.log(dim))
        self.f = 0.5  # Differential weight
        self.cr = 0.9  # Crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.used_budget = 0
        self.sigma = 0.3
        self.mean = np.mean(self.population, axis=0)
        self.covariance = np.identity(dim)

    def __call__(self, func):
        while self.used_budget < self.budget:
            # Evaluate fitness of population
            for i in range(self.population_size):
                if self.fitness[i] == np.inf:
                    self.fitness[i] = func(self.population[i])
                    self.used_budget += 1
                    if self.used_budget >= self.budget:
                        return self._best_solution()

            # Differential Evolution step
            for i in range(self.population_size):
                a, b, c = self._select_random_indices(i)
                mutant = self.population[a] + self.f * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                trial = np.copy(self.population[i])
                if np.random.rand() < self.cr:
                    trial = mutant

                trial_fitness = func(trial)
                self.used_budget += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                if self.used_budget >= self.budget:
                    return self._best_solution()

            # CMA-ES step
            self.mean = np.mean(self.population, axis=0)
            self.covariance = (1 - self.sigma) * self.covariance + self.sigma * np.cov(self.population.T)
            if self.used_budget < self.budget:
                new_samples = np.random.multivariate_normal(self.mean, self.covariance, self.population_size)
                new_samples = np.clip(new_samples, self.lower_bound, self.upper_bound)
                for i in range(self.population_size):
                    new_fitness = func(new_samples[i])
                    self.used_budget += 1
                    if new_fitness < self.fitness[i]:
                        self.population[i] = new_samples[i]
                        self.fitness[i] = new_fitness
                    if self.used_budget >= self.budget:
                        return self._best_solution()

        return self._best_solution()

    def _select_random_indices(self, exclude_index):
        indices = list(range(self.population_size))
        indices.remove(exclude_index)
        return np.random.choice(indices, 3, replace=False)

    def _best_solution(self):
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]