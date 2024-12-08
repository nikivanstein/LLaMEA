import numpy as np
import random

class HSABC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.HMS = 10
        self.HMCR = 0.5
        self.PAR = 0.5
        self.bee_population_size = 50
        self.x = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.inf * np.ones(self.population_size)
        self.best_x = np.inf * np.ones(self.dim)
        self.best_fitness = np.inf
        self.bee_x = np.random.uniform(self.lower_bound, self.upper_bound, (self.bee_population_size, self.dim))
        self.bee_fitness = np.inf * np.ones(self.bee_population_size)
        self.best_bee_x = np.inf * np.ones(self.dim)
        self.best_bee_fitness = np.inf

    def __call__(self, func):
        for i in range(self.budget):
            y = func(self.x)
            self.fitness = y
            idx = np.argmin(y)
            self.best_x = self.x[idx]
            self.best_fitness = y[idx]
            for j in range(self.population_size):
                if j!= idx:
                    r1, r2 = random.sample(range(self.population_size), 2)
                    while r1 == idx or r2 == idx:
                        r1, r2 = random.sample(range(self.population_size), 2)
                    x_new = self.x[r1] + (self.x[r2] - self.x[idx]) * np.random.uniform(0, 1)
                    x_new = x_new + np.random.uniform(-1, 1, self.dim)
                    x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                    y_new = func(x_new)
                    if y_new < self.fitness[j]:
                        self.x[j] = x_new
                        self.fitness[j] = y_new
            self.HMCR = self.HMCR + 0.01 * (self.HMCR - 0.5)
            self.HMCR = max(0.1, min(1.0, self.HMCR))
            self.PAR = self.PAR + 0.01 * (self.PAR - 0.5)
            self.PAR = max(0.1, min(1.0, self.PAR))
            self.HMS = int(self.population_size * self.HMCR)
            self.bee_x = np.random.uniform(self.lower_bound, self.upper_bound, (self.bee_population_size, self.dim))
            self.bee_fitness = func(self.bee_x)
            for j in range(self.bee_population_size):
                r1, r2 = random.sample(range(self.bee_population_size), 2)
                while r1 == j or r2 == j:
                    r1, r2 = random.sample(range(self.bee_population_size), 2)
                bee_x_new = self.bee_x[r1] + (self.bee_x[r2] - self.bee_x[j]) * np.random.uniform(0, 1)
                bee_x_new = bee_x_new + np.random.uniform(-1, 1, self.dim)
                bee_x_new = np.clip(bee_x_new, self.lower_bound, self.upper_bound)
                y_new = func(bee_x_new)
                if y_new < self.bee_fitness[j]:
                    self.bee_x[j] = bee_x_new
                    self.bee_fitness[j] = y_new
            idx = np.argmin(self.bee_fitness)
            self.best_bee_x = self.bee_x[idx]
            self.best_bee_fitness = self.bee_fitness[idx]
            if self.best_bee_fitness < self.best_fitness:
                self.best_fitness = self.best_bee_fitness
                self.best_x = self.best_bee_x
        return self.best_x, self.best_fitness