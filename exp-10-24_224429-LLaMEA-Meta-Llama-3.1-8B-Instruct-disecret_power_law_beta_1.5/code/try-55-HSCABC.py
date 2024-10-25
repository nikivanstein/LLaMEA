import numpy as np
import random

class HSCABC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.harmmony_memory_size = 20
        self.HS_max_iter = 20
        self.HS_p = 0.95
        self.ABC_max_iter = 20
        self.ABC_limit = 0.01
        self.x = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.inf * np.ones(self.population_size)
        self.best_x = np.inf * np.ones(self.dim)
        self.best_fitness = np.inf
        self.harmony_memory = np.zeros((self.harmmony_memory_size, self.dim))

    def __call__(self, func):
        for i in range(self.budget):
            y = func(self.x)
            self.fitness = y
            idx = np.argmin(y)
            self.best_x = self.x[idx]
            self.best_fitness = y[idx]
            for j in range(self.population_size):
                if j!= idx:
                    r1, r2, r3 = random.sample(range(self.population_size), 3)
                    while r1 == idx or r2 == idx or r3 == idx:
                        r1, r2, r3 = random.sample(range(self.population_size), 3)
                    x_new = self.x[r1] + self.x[r2] - self.x[r3]
                    x_new = x_new + np.random.uniform(-1, 1, self.dim)
                    x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                    y_new = func(x_new)
                    if y_new < self.fitness[j]:
                        self.x[j] = x_new
                        self.fitness[j] = y_new
            # Harmony Search
            for k in range(self.HS_max_iter):
                for j in range(self.population_size):
                    if random.random() < self.HS_p:
                        x_new = self.harmony_memory[np.random.randint(0, self.harmmony_memory_size), :]
                        x_new = x_new + np.random.uniform(-1, 1, self.dim)
                        x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                        y_new = func(x_new)
                        if y_new < self.fitness[j]:
                            self.x[j] = x_new
                            self.fitness[j] = y_new
                # Update Harmony Memory
                self.harmony_memory[np.argmin(self.fitness)] = self.x[np.argmin(self.fitness)]
            # Artificial Bee Colony
            for k in range(self.ABC_max_iter):
                for j in range(self.population_size):
                    if random.random() < self.ABC_limit:
                        r1, r2, r3 = random.sample(range(self.population_size), 3)
                        while r1 == j or r2 == j or r3 == j:
                            r1, r2, r3 = random.sample(range(self.population_size), 3)
                        x_new = self.x[r1] + self.x[r2] - self.x[r3]
                        x_new = x_new + np.random.uniform(-1, 1, self.dim)
                        x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                        y_new = func(x_new)
                        if y_new < self.fitness[j]:
                            self.x[j] = x_new
                            self.fitness[j] = y_new
            self.best_fitness = np.min(self.fitness)
            self.best_x = self.x[np.argmin(self.fitness)]
        return self.best_x, self.best_fitness