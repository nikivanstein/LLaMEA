import numpy as np

class AQIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.best_personal_positions = np.copy(self.population)
        self.best_personal_fitness = np.full(self.pop_size, float('inf'))
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0
        self.c1 = 2.0  # cognitive parameter
        self.c2 = 2.0  # social parameter
        self.w = 0.5   # inertia weight

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def adaptive_inertia(self):
        return 0.5 + np.random.rand() * 0.5

    def __call__(self, func):
        np.random.seed(42)

        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])
            if self.fitness[i] < self.best_personal_fitness[i]:
                self.best_personal_fitness[i] = self.fitness[i]
                self.best_personal_positions[i] = self.population[i]
            if self.fitness[i] < self.best_global_fitness:
                self.best_global_fitness = self.fitness[i]
                self.best_global_position = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # Update velocity
                self.velocities[i] = (self.w * self.velocities[i] +
                                     self.c1 * r1 * (self.best_personal_positions[i] - self.population[i]) +
                                     self.c2 * r2 * (self.best_global_position - self.population[i]))
                
                # Update position
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

                # Evaluate new position
                current_fitness = self.evaluate(func, self.population[i])
                if current_fitness < self.best_personal_fitness[i]:
                    self.best_personal_fitness[i] = current_fitness
                    self.best_personal_positions[i] = self.population[i]
                if current_fitness < self.best_global_fitness:
                    self.best_global_fitness = current_fitness
                    self.best_global_position = self.population[i]

            # Adapt inertia weight
            self.w = self.adaptive_inertia()

        return self.best_global_position