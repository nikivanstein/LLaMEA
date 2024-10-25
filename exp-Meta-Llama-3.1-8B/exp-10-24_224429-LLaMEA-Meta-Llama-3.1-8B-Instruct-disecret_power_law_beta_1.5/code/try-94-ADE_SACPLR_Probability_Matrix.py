import numpy as np
import random

class ADE_SACPLR_Probability_Matrix:
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
        self.mutation_probability_matrix = np.random.uniform(0, 1, (self.population_size, self.dim))
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
            self.mutation_probability_matrix = np.clip(self.mutation_probability_matrix + self.learning_rate * (np.random.uniform(0, 1, (self.population_size, self.dim)) - self.mutation_probability_matrix), 0, 1)
            for k in range(self.dim):
                idx = np.argmin(self.fitness)
                if random.random() < self.mutation_probability_matrix[idx, k]:
                    self.x[idx, k] = self.x[idx, k] + self.sigma * np.random.normal(0, 1)
                    self.x[idx, k] = np.clip(self.x[idx, k], self.lower_bound, self.upper_bound)
                    y_new = func(self.x[idx])
                    if y_new < self.fitness[idx]:
                        self.fitness[idx] = y_new
            if self.fitness[idx] < self.best_fitness:
                self.best_fitness = self.fitness[idx]
                self.best_x = self.x[idx]
        return self.best_x, self.best_fitness