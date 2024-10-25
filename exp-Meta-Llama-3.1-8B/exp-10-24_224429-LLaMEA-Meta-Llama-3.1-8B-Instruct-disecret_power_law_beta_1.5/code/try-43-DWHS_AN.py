import numpy as np
import random

class DWHS_AN:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.harmony_memory_size = 10
        self.panning_rate = 0.01
        self.weight_update_rate = 0.1
        self.neighborhood_size = 5
        self.x = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.inf * np.ones(self.population_size)
        self.best_x = np.inf * np.ones(self.dim)
        self.best_fitness = np.inf
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))

    def __call__(self, func):
        for i in range(self.budget):
            y = func(self.x)
            self.fitness = y
            idx = np.argmin(y)
            self.best_x = self.x[idx]
            self.best_fitness = y[idx]
            for j in range(self.population_size):
                if j!= idx:
                    r1 = random.randint(0, self.population_size - 1)
                    while r1 == idx:
                        r1 = random.randint(0, self.population_size - 1)
                    x_new = self.harmony_memory[random.randint(0, self.harmony_memory_size - 1)]
                    x_new = x_new + np.random.normal(0, 1, self.dim) * self.panning_rate
                    x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                    y_new = func(x_new)
                    if y_new < self.fitness[j]:
                        self.x[j] = x_new
                        self.fitness[j] = y_new
            self.harmony_memory = np.clip(self.harmony_memory + np.random.normal(0, self.panning_rate, (self.harmony_memory_size, self.dim)), self.lower_bound, self.upper_bound)
            self.weight_update_rate = self.weight_update_rate * 0.9
            self.neighborhood_size = int(self.population_size * (1 - self.weight_update_rate))
            self.x[idx] = self.harmony_memory[random.randint(0, self.harmony_memory_size - 1)]
            self.fitness[idx] = func(self.x[idx])
            if self.fitness[idx] < self.best_fitness:
                self.best_fitness = self.fitness[idx]
                self.best_x = self.x[idx]
        return self.best_x, self.best_fitness