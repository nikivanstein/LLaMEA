import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Common heuristic for population size
        self.bounds = (-5.0, 5.0)
        self.F = 0.5  # Initial mutation factor
        self.CR = 0.9  # Crossover probability
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.used_budget = 0
        self.adaptive_population = True  # Adaptive population size

    def __call__(self, func):
        # Evaluate initial population
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            self.used_budget += 1
            if self.used_budget >= self.budget:
                break

        while self.used_budget < self.budget:
            if self.adaptive_population:
                self.population_size = max(4, int((self.budget - self.used_budget) / 2))  # Adaptive reduction
                self.population = self.population[:self.population_size]
                self.fitness = self.fitness[:self.population_size]
            for i in range(self.population_size):
                # Mutation
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c) + 0.1 * (np.mean(self.population, axis=0) - a), self.bounds[0], self.bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.used_budget += 1
                if f_trial < self.fitness[i]:
                    self.fitness[i] = f_trial
                    self.population[i] = trial
                
                if self.used_budget >= self.budget:
                    break

            # Adaptive mutation factor update
            self.F = 0.5 + (0.5 * np.random.rand())

        # Return the best found solution
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]