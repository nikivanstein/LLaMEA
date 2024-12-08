import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.bounds = (-5.0, 5.0)
        self.F = 0.5
        self.CR = 0.9
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.used_budget = 0
        self.successful_mutations = 0

    def __call__(self, func):
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            self.used_budget += 1
            if self.used_budget >= self.budget:
                break

        while self.used_budget < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])

                f_trial = func(trial)
                self.used_budget += 1
                if f_trial < self.fitness[i]:
                    self.fitness[i] = f_trial
                    self.population[i] = trial
                    self.successful_mutations += 1

                if self.used_budget >= self.budget:
                    break

            self.F = 0.4 + 0.6 * (self.successful_mutations / max(1, self.population_size))
            self.CR = 0.9 - 0.5 * (self.successful_mutations / max(1, self.population_size))
            self.successful_mutations = 0

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]