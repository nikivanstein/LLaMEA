import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.evaluations = 0

    def __call__(self, func):
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Mutation
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.evaluations += 1

                if f_trial < self.fitness[i]:
                    self.fitness[i] = f_trial
                    self.population[i] = trial
                else:
                    # Adjust F and CR if no improvement
                    self.F = max(0.1, self.F * 0.99)
                    self.CR = min(0.9, self.CR * 1.01)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]