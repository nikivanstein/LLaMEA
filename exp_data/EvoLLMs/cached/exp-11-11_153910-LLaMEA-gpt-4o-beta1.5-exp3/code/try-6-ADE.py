import numpy as np

class ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 40
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def __call__(self, func):
        np.random.seed(42)
        F = 0.5  # Differential weight
        CR = 0.9  # Crossover probability

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                indices = [index for index in range(self.pop_size) if index != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)

                crossover_vector = np.random.rand(self.dim) < CR
                trial_vector = np.where(crossover_vector, mutant_vector, self.population[i])

                trial_fitness = self.evaluate(func, trial_vector)
                current_fitness = self.evaluate(func, self.population[i])

                if trial_fitness < current_fitness:
                    self.population[i] = trial_vector
                    current_fitness = trial_fitness

                if current_fitness < self.best_global_fitness:
                    self.best_global_fitness = current_fitness
                    self.best_global_position = self.population[i]

            # Adaptive adjustment of F and CR based on progress
            if np.random.rand() < 0.1:
                F = 0.5 + 0.3 * np.sin(self.evaluations * np.pi / self.budget)
                CR = 0.9 * np.cos(self.evaluations * np.pi / self.budget) + 0.1

        return self.best_global_position