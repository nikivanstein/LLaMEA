# Description: Refined Adaptive Differential Evolution by dynamically adjusting differential weight and adapting population size more effectively.
# Code: 
import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 8 * dim  # Adjusted initial population size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, dim))
        self.fitness = np.full(self.initial_population_size, np.inf)
        self.F = 0.7  # Adjusted differential weight for initial exploration
        self.CR = 0.8  # Adjusted initial crossover probability
        self.evaluations = 0

    def __call__(self, func):
        population_size = self.initial_population_size
        for i in range(population_size):
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1

        while self.evaluations < self.budget:
            for i in range(population_size):
                if self.evaluations >= self.budget:
                    break

                # Mutation
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                
                # Enhanced dynamic adaptation of differential weight
                self.F = 0.4 + 0.3 * np.random.rand() + (0.3 * (1 - self.evaluations / self.budget))

                # Adaptive crossover probability
                self.CR = 0.7 + (0.2 * (1 - self.evaluations / self.budget))
                
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

            # More gradual reduction in population size
            if self.evaluations > self.budget * 0.5 and population_size > 5:
                population_size = max(5, int(population_size * 0.75))
                self.population = self.population[:population_size]
                self.fitness = self.fitness[:population_size]

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]