import numpy as np
import random

class HPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.max_iter = 100
        self.w = 0.7298
        self.c1 = 1.49618
        self.c2 = 1.49618
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = float('inf')
        self.selected_solution = np.random.choice(self.population_size, 1, replace=False)[0]
        self.selected_solution_fitness = float('inf')

    def __call__(self, func):
        for i in range(self.max_iter):
            for j in range(self.population_size):
                x = np.copy(self.population[j])
                if random.random() < 0.05:
                    self.population[j] = self.population[self.selected_solution] + np.random.uniform(-1, 1, self.dim)
                    self.population[j] = np.clip(self.population[j], -5.0, 5.0)
                else:
                    r1 = random.randint(0, self.population_size - 1)
                    r2 = random.randint(0, self.population_size - 1)
                    r3 = random.randint(0, self.population_size - 1)
                    x = self.population[r1] + self.w * (self.population[r2] - self.population[r3]) + np.random.uniform(-1, 1, self.dim)
                    x = np.clip(x, -5.0, 5.0)
                fitness = func(x)
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = x
                if fitness < self.selected_solution_fitness:
                    self.selected_solution_fitness = fitness
                    self.selected_solution = j
            self.population_size = int(self.population_size * (1 - (i / self.max_iter)))
            self.population = np.random.choice(self.population, self.population_size, replace=False)
            if i % 10 == 0:
                print(f"Iteration {i+1}, Best fitness: {self.best_fitness}")
        return self.best_solution