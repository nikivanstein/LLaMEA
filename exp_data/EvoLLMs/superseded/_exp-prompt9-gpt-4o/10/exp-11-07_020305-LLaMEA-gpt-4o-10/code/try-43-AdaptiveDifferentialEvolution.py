import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 8 * dim  # Initial population size remains the same
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.F = 0.6  # Revised initial differential weight for balanced exploration
        self.CR = 0.9  # Increased crossover probability for enhanced variability
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
                
                # Dynamic adaptation of differential weight
                self.F = 0.5 + 0.2 * np.random.rand() + (0.3 * (1 - self.evaluations / self.budget))

                # Adaptive crossover probability
                self.CR = 0.8 + (0.1 * (1 - self.evaluations / self.budget))
                
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

            # Gradual reduction in population size
            if self.evaluations > self.budget * 0.5 and self.population_size > 5:
                self.population_size = max(5, int(self.population_size * 0.8))
                self.population = self.population[:self.population_size]
                self.fitness = self.fitness[:self.population_size]

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]