import numpy as np
import random

class ADE_MPLR_SACPLR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.F = 0.5
        self.CR = 0.5
        self.sigma = 0.1
        self.learning_rate_1 = 0.01
        self.learning_rate_2 = 0.001
        self.crossover_probability = 0.5
        self.x = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.inf * np.ones(self.population_size)
        self.best_x = np.inf * np.ones(self.dim)
        self.best_fitness = np.inf
        self.phase = 0
        self.phase_length = int(self.budget * 0.06521739130434782)

    def __call__(self, func):
        for i in range(self.budget):
            y = func(self.x)
            self.fitness = y
            idx = np.argmin(y)
            self.best_x = self.x[idx]
            self.best_fitness = y[idx]
            if self.phase < self.phase_length:
                self.sigma = self.sigma + self.learning_rate_1 * (self.sigma - self.fitness[idx])
                self.CR = self.CR + self.learning_rate_1 * (self.crossover_probability - self.CR)
                self.crossover_probability = max(0.1, min(1.0, self.CR))
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
                self.phase += 1
            else:
                self.sigma = self.sigma + self.learning_rate_2 * (self.sigma - self.fitness[idx])
                self.CR = self.CR + self.learning_rate_2 * (self.crossover_probability - self.CR)
                self.crossover_probability = max(0.1, min(1.0, self.CR))
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
        return self.best_x, self.best_fitness