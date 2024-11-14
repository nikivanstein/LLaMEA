import numpy as np

class OptimizedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 8 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.F = 0.6
        self.CR = 0.9
        self.evaluations = 0

    def __call__(self, func):
        for i in range(self.pop_size):
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break

                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)

                self.F = 0.5 + 0.35 * np.random.rand() + 0.2 * (1 - self.evaluations / self.budget)
                self.CR = 0.7 + 0.2 * (1 - self.evaluations / self.budget)

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, self.population[i])

                f_trial = func(trial)
                self.evaluations += 1

                if f_trial < self.fitness[i]:
                    self.fitness[i] = f_trial
                    self.population[i] = trial

            if self.evaluations > self.budget * 0.7 and self.pop_size > 5:
                self.pop_size = max(5, int(self.pop_size * 0.75))
                self.population = self.population[:self.pop_size]
                self.fitness = self.fitness[:self.pop_size]

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]