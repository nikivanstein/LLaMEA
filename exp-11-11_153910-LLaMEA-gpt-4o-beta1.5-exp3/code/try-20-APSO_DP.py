import numpy as np

class APSO_DP:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 40
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_fitness = np.full(self.pop_size, float('inf'))
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0
        self.w = 0.5  # Inertia weight
        self.c1 = 2.0  # Cognitive acceleration coefficient
        self.c2 = 2.0  # Social acceleration coefficient
        self.F = 0.5  # Differential perturbation factor

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def __call__(self, func):
        np.random.seed(42)

        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])
            self.personal_best_positions[i] = np.copy(self.population[i])
            self.personal_best_fitness[i] = self.fitness[i]
            if self.fitness[i] < self.best_global_fitness:
                self.best_global_fitness = self.fitness[i]
                self.best_global_position = np.copy(self.population[i])

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
                social = self.c2 * r2 * (self.best_global_position - self.population[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social

                # Adaptive velocity control
                if np.random.rand() < 0.5:
                    indices = [idx for idx in range(self.pop_size) if idx != i]
                    x1, x2 = self.population[np.random.choice(indices, 2, replace=False)]
                    perturbation = self.F * (x1 - x2)
                    self.velocities[i] += perturbation

                # Update position
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

                # Evaluate new position
                fitness_value = self.evaluate(func, self.population[i])
                if fitness_value < self.personal_best_fitness[i]:
                    self.personal_best_positions[i] = np.copy(self.population[i])
                    self.personal_best_fitness[i] = fitness_value
                if fitness_value < self.best_global_fitness:
                    self.best_global_fitness = fitness_value
                    self.best_global_position = np.copy(self.population[i])

        return self.best_global_position