import numpy as np

class DE_CMA_ES_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 4 + int(3 * np.log(dim))
        self.f = 0.6
        self.cr = 0.8
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.used_budget = 0
        self.sigma = 0.2
        self.mean = np.mean(self.population, axis=0)
        self.covariance = np.identity(dim)

    def __call__(self, func):
        while self.used_budget < self.budget:
            not_evaluated = np.isinf(self.fitness)
            if np.any(not_evaluated):
                self.fitness[not_evaluated] = [func(ind) for ind in self.population[not_evaluated]]
                self.used_budget += np.sum(not_evaluated)
                if self.used_budget >= self.budget:
                    return self._best_solution()
            
            for i in range(self.population_size):
                a, b, c = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                mutant = self.population[a] + self.f * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                mask = np.random.rand(self.dim) < self.cr
                trial = np.where(mask, mutant, self.population[i])

                trial_fitness = func(trial)
                self.used_budget += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                if self.used_budget >= self.budget:
                    return self._best_solution()

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

    def _best_solution(self):
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]