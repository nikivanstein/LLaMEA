import numpy as np

class ADEDSOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.scale_factor = 0.5
        self.crossover_rate = 0.7
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.eval_count = 0

    def __call__(self, func):
        while self.eval_count < self.budget:
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break

                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[indices]
                mutant = a + self.scale_factor * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.where(np.random.rand(self.dim) <= self.crossover_rate, mutant, self.population[i])

                # Evaluate
                trial_fitness = func(trial)
                self.eval_count += 1

                # Selection
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

            # Dynamic adaptation of parameters
            self.scale_factor = 0.5 + 0.5 * (1 - self.eval_count / self.budget)
            self.crossover_rate = 0.1 + 0.9 * (self.eval_count / self.budget)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]