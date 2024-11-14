import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.bounds = (-5.0, 5.0)
        self.F = 0.5
        self.CR = 0.9
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.initial_population_size, dim))
        self.fitness = np.full(self.initial_population_size, np.inf)
        self.used_budget = 0
        self.population_size = self.initial_population_size

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
                
                if self.used_budget >= self.budget:
                    break

            self.F = 0.5 + (0.3 * np.random.rand())  # Adaptive reduction
            self.CR = 0.9 - (0.1 * np.random.rand())  # Adaptive crossover

            # Dynamic population resizing
            if self.used_budget < self.budget / 2 and self.population_size > dim:
                self.population_size = max(dim, int(self.population_size * 0.9))
                self.population = self.population[:self.population_size]
                self.fitness = self.fitness[:self.population_size]

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]