import numpy as np
import random

class DEAMP:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = 10
        self.max_iter = 100
        self.population_size = 50
        self.w = 0.7298
        self.c1 = 1.49618
        self.c2 = 1.49618
        self.memory = np.random.uniform(-5.0, 5.0, (self.memory_size, self.dim))
        self.best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = float('inf')
        self.selected_solution = np.random.randint(0, self.population_size)

    def __call__(self, func):
        for i in range(self.max_iter):
            for j in range(self.population_size):
                if j == self.selected_solution and random.random() < 0.05:
                    x = self.memory[j] + np.random.uniform(-1, 1, self.dim)
                else:
                    x = np.random.uniform(-5.0, 5.0, self.dim)
                for k in range(self.dim):
                    r1 = random.random()
                    r2 = random.random()
                    if r1 < 0.5:
                        x[k] = self.memory[np.random.randint(0, self.memory_size), k] + self.w * (self.memory[np.random.randint(0, self.memory_size), k] - self.memory[np.random.randint(0, self.memory_size), k])
                    else:
                        x[k] = self.best_solution[k] + self.w * (self.best_solution[k] - self.memory[np.random.randint(0, self.memory_size), k])
                fitness = func(x)
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = x
                self.memory[j] = x
            self.memory_size = int(self.memory_size * (1 - (i / self.max_iter)))
            self.memory = np.random.choice(self.memory, self.memory_size, replace=False)
            if i % 10 == 0:
                print(f"Iteration {i+1}, Best fitness: {self.best_fitness}")
        return self.best_solution