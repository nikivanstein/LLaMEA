import numpy as np
import random

class DECS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.max_iter = 100
        self.f = 0.5
        self.a = 0.01
        self.b = 0.01
        self.x = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.y = np.copy(self.x)
        self.best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = float('inf')

    def __call__(self, func):
        for i in range(self.max_iter):
            for j in range(self.pop_size):
                r1 = random.random()
                r2 = random.random()
                if r1 < 0.05:
                    x = self.y[j]
                    for k in range(self.dim):
                        x[k] = self.x[np.random.randint(0, self.pop_size), k] + self.f * (self.x[np.random.randint(0, self.pop_size), k] - self.x[np.random.randint(0, self.pop_size), k])
                else:
                    x = self.x[j]
                    v1 = self.x[np.random.randint(0, self.pop_size), np.random.randint(0, self.dim)]
                    v2 = self.x[np.random.randint(0, self.pop_size), np.random.randint(0, self.dim)]
                    for k in range(self.dim):
                        x[k] = v1[k] + self.a * (v1[k] - v2[k])
                fitness = func(x)
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = x
                self.y[j] = x
            self.x = np.copy(self.y)
            if i % 10 == 0:
                print(f"Iteration {i+1}, Best fitness: {self.best_fitness}")
        return self.best_solution