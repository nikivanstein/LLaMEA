import numpy as np
import random

class DE_CMAES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.max_iter = 100
        self.cr = 0.5
        self.c1 = 0.5
        self.c2 = 0.5
        self.mutation_factor = 1.0
        self.step_size = 0.1
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = float('inf')
        self.probability = 0.05

    def __call__(self, func):
        for i in range(self.max_iter):
            for j in range(self.pop_size):
                x = self.population[j]
                if random.random() < self.probability:
                    for k in range(self.dim):
                        r1, r2, r3 = np.random.randint(0, self.pop_size, 3)
                        x[k] = self.population[r1, k] + self.mutation_factor * (self.population[r2, k] - self.population[r3, k])
                else:
                    r1, r2 = np.random.randint(0, self.pop_size, 2)
                    x = self.population[r1] + self.c1 * (self.population[r2] - self.population[np.random.randint(0, self.pop_size)])
                x = x + self.step_size * np.random.uniform(-5.0, 5.0, self.dim)
                x = np.clip(x, -5.0, 5.0)
                fitness = func(x)
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = x
                self.population[j] = x
            self.mutation_factor = self.mutation_factor + (random.random() - 0.5) * 0.1
            self.step_size = self.step_size + (random.random() - 0.5) * 0.01
            self.probability = self.probability + (random.random() - 0.5) * 0.01
            if i % 10 == 0:
                print(f"Iteration {i+1}, Best fitness: {self.best_fitness}")
        return self.best_solution