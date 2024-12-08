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

                # Mutation with dynamic scale factor adjustment
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[indices]
                dynamic_factor = np.abs(np.random.normal(self.scale_factor, 0.1))
                mutant = a + dynamic_factor * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover with adaptive probability
                crossover_points = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover_points, mutant, self.population[i])

                # Evaluate
                trial_fitness = func(trial)
                self.eval_count += 1

                # Selection
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

            # Dynamic adaptation of parameters
            self.scale_factor = 0.4 + 0.6 * (1 - self.eval_count / self.budget)
            self.crossover_rate = 0.2 + 0.8 * (self.eval_count / self.budget)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]