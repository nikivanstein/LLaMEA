import numpy as np
import random

class SADEELM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.F = 0.5
        self.CR = 0.5
        self.sigma = 0.1
        self.learning_rate = 0.01
        self.crossover_probability = 0.5
        self.memory_size = 10
        self.memory = np.zeros((self.memory_size, self.dim))
        self.memory_fitness = np.inf * np.ones(self.memory_size)
        self.x = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.inf * np.ones(self.population_size)
        self.best_x = np.inf * np.ones(self.dim)
        self.best_fitness = np.inf

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
                    x_new = self.x[r1] + self.F * (self.x[r2] - self.x[r3])
                    x_new = x_new + self.sigma * np.random.normal(0, 1, self.dim)
                    x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                    y_new = func(x_new)
                    if y_new < self.fitness[j]:
                        self.x[j] = x_new
                        self.fitness[j] = y_new
            self.CR = self.CR + self.learning_rate * (self.crossover_probability - self.CR)
            self.crossover_probability = max(0.1, min(1.0, self.CR))
            self.sigma = self.sigma + self.learning_rate * (self.sigma - self.fitness[idx])
            if self.fitness[idx] < self.best_fitness:
                self.best_fitness = self.fitness[idx]
                self.best_x = self.x[idx]
            # Ensemble learning and memory update
            if i % 10 == 0 and i > 0:
                self.memory[i % self.memory_size, :] = self.best_x
                self.memory_fitness[i % self.memory_size] = self.best_fitness
                self.memory_fitness = np.sort(self.memory_fitness)
                self.memory = self.memory[self.memory_fitness < self.memory_fitness[-1]]
                self.memory = self.memory[:self.memory_size]
                self.memory_fitness = self.memory_fitness[:self.memory_size]
                self.x = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
                for j in range(self.population_size):
                    x_new = np.zeros(self.dim)
                    for k in range(self.memory_size):
                        x_new += self.memory[k] * np.random.uniform(0, 1)
                    x_new = x_new / self.memory_size
                    x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                    self.x[j] = x_new
                    y_new = func(self.x[j])
                    self.fitness[j] = y_new
        return self.best_x, self.best_fitness