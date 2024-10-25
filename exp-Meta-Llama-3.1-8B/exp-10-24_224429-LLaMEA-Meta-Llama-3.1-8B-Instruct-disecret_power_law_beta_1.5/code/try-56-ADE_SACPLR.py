import numpy as np
import random

class ADE_SACPLR:
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
                    x_new = x_new + self.sigma * np.random levy(self.dim)
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
            # Probability-driven update
            prob = 0.021739130434782608
            if random.random() < prob:
                for k in range(self.dim):
                    if random.random() < 0.5:
                        self.x[idx, k] = self.x[idx, k] + self.F * (self.x[r1, k] - self.x[r2, k])
                    else:
                        self.x[idx, k] = self.x[idx, k] + self.F * (self.x[r2, k] - self.x[r1, k])
        return self.best_x, self.best_fitness

def np_random_levy(dim):
    u = np.random.normal(0, 1, dim)
    v = np.random.normal(0, 1, dim)
    s = np.sum(u**2)
    while s <= 0:
        u = np.random.normal(0, 1, dim)
        s = np.sum(u**2)
    t = np.sum(v**2) / s
    f = np.sqrt(t) / (1 + t)
    g = np.sqrt(t) * u
    return f * g