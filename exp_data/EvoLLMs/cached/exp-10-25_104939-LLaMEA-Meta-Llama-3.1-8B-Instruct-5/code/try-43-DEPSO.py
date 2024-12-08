import numpy as np
import random

class DEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.max_iter = 100
        self.swarm_size = 50
        self.w = 0.7298
        self.c1 = 1.49618
        self.c2 = 1.49618
        self.F = 0.5
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = float('inf')
        self.selected_solution_index = np.random.randint(0, self.population_size)

    def __call__(self, func):
        for i in range(self.max_iter):
            for j in range(self.population_size):
                if random.random() < 0.05:
                    self.population[j] = self.population[self.selected_solution_index]
                x = self.population[j] + self.F * (self.population[np.random.randint(0, self.population_size), :] - self.population[np.random.randint(0, self.population_size), :])
                for k in range(self.dim):
                    r1 = random.random()
                    r2 = random.random()
                    if r1 < 0.5:
                        x[k] = self.population[j, k] + self.w * (self.population[j, k] - self.population[np.random.randint(0, self.population_size), k])
                    else:
                        x[k] = self.best_solution[k] + self.w * (self.best_solution[k] - self.population[np.random.randint(0, self.population_size), k])
                fitness = func(x)
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = x
                self.population[j] = x
            self.selected_solution_index = np.random.randint(0, self.population_size)
            if i % 10 == 0:
                print(f"Iteration {i+1}, Best fitness: {self.best_fitness}")
            if self.population_size > 10 and self.best_fitness < 0.1:
                self.population_size = int(self.population_size * 0.9)
                self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        return self.best_solution