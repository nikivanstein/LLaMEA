import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, dim))
        self.fitness = np.full(self.initial_population_size, np.inf)
        self.F = 0.8
        self.CR = 0.9
        self.evaluations = 0

    def __call__(self, func):
        for i in range(self.initial_population_size):
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1

        while self.evaluations < self.budget:
            for i in range(len(self.population)):
                if self.evaluations >= self.budget:
                    break

                idxs = np.random.choice([idx for idx in range(self.initial_population_size) if idx != i], 3, replace=False)
                a, b, c = self.population[idxs]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)

                self.F = 0.5 + 0.3 * np.random.randn()  # Adjusted stochastic factor
                self.CR = 0.9 * (1.0 - self.evaluations / self.budget)

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, self.population[i])

                f_trial = func(trial)
                self.evaluations += 1

                if f_trial < self.fitness[i]:
                    self.fitness[i] = f_trial
                    self.population[i] = trial

            if self.evaluations > self.budget * 0.5:
                self.population = self.population[:len(self.population) // 2]
                self.fitness = self.fitness[:len(self.fitness) // 2]

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]