import numpy as np
import random

class AMSHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.harmony_memory_size = int(0.021739130434782608 * self.population_size)
        self.F = 0.5
        self.CR = 0.5
        self.sigma = 0.1
        self.learning_rate = 0.01
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.fitness = np.inf * np.ones(self.harmony_memory_size)
        self.best_x = np.inf * np.ones(self.dim)
        self.best_fitness = np.inf

    def __call__(self, func):
        for i in range(self.budget):
            y = func(self.harmony_memory)
            self.fitness = y
            idx = np.argmin(y)
            self.best_x = self.harmony_memory[idx]
            self.best_fitness = y[idx]
            for j in range(self.harmony_memory_size):
                r1, r2 = random.sample(range(self.harmony_memory_size), 2)
                while r1 == idx or r2 == idx:
                    r1, r2 = random.sample(range(self.harmony_memory_size), 2)
                x_new = self.harmony_memory[r1] + self.F * (self.harmony_memory[r2] - self.best_x)
                x_new = x_new + self.sigma * np.random.normal(0, 1, self.dim)
                x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                y_new = func(x_new)
                if y_new < self.fitness[j]:
                    self.harmony_memory[j] = x_new
                    self.fitness[j] = y_new
            self.harmony_memory_size = int(0.021739130434782608 * self.population_size)
            self.harmony_memory = np.concatenate((self.harmony_memory[:self.harmony_memory_size], [x_new for x_new in self.harmony_memory]))
            self.fitness = np.concatenate((self.fitness[:self.harmony_memory_size], [y_new for y_new in self.fitness]))
            self.CR = self.CR + self.learning_rate * (self.CR - self.fitness[idx])
            self.crossover_probability = max(0.1, min(1.0, self.CR))
            self.sigma = self.sigma + self.learning_rate * (self.sigma - self.fitness[idx])
            if self.fitness[idx] < self.best_fitness:
                self.best_fitness = self.fitness[idx]
                self.best_x = self.harmony_memory[idx]
        return self.best_x, self.best_fitness