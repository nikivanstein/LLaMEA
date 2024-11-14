import numpy as np

class QPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.best_personal_positions = np.copy(self.population)
        self.best_personal_fitness = np.copy(self.fitness)
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0
        self.alpha = 0.75  # Balance coefficient

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def update_positions(self):
        for i in range(self.pop_size):
            p = self.best_personal_positions[i]
            g = self.best_global_position
            u = np.random.rand(self.dim)
            v = np.random.rand(self.dim)
            delta = self.alpha * np.abs(p - g) * np.log(1 / u)
            new_position = (p + g) / 2 + np.sign(u - 0.5) * delta * v
            new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
            self.population[i] = new_position

    def __call__(self, func):
        np.random.seed(42)

        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])
            self.best_personal_fitness[i] = self.fitness[i]
            if self.fitness[i] < self.best_global_fitness:
                self.best_global_fitness = self.fitness[i]
                self.best_global_position = self.population[i]

        while self.evaluations < self.budget:
            self.update_positions()
            for i in range(self.pop_size):
                current_fitness = self.evaluate(func, self.population[i])
                if current_fitness < self.best_personal_fitness[i]:
                    self.best_personal_fitness[i] = current_fitness
                    self.best_personal_positions[i] = self.population[i]
                if current_fitness < self.best_global_fitness:
                    self.best_global_fitness = current_fitness
                    self.best_global_position = self.population[i]

        return self.best_global_position