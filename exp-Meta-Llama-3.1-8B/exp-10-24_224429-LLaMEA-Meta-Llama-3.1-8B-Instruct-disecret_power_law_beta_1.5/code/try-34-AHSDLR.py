import numpy as np
import random

class AHSDLR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.HMCR = 0.9
        self.PAR = 0.1
        self.F = 0.5
        self.CR = 0.5
        self.sigma = 0.1
        self.learning_rate = 0.01
        self.x = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.inf * np.ones(self.population_size)
        self.best_x = np.inf * np.ones(self.dim)
        self.best_fitness = np.inf
        self.HMCR_history = np.zeros(self.budget)
        self.PAR_history = np.zeros(self.budget)
        self.F_history = np.zeros(self.budget)
        self.CR_history = np.zeros(self.budget)
        self.sigma_history = np.zeros(self.budget)

    def __call__(self, func):
        for i in range(self.budget):
            y = func(self.x)
            self.fitness = y
            idx = np.argmin(y)
            self.best_x = self.x[idx]
            self.best_fitness = y[idx]
            self.HMCR_history[i] = self.HMCR
            self.PAR_history[i] = self.PAR
            self.F_history[i] = self.F
            self.CR_history[i] = self.CR
            self.sigma_history[i] = self.sigma
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
            self.HMCR = self.HMCR + self.learning_rate * (self.HMCR - np.mean(self.fitness))
            self.PAR = self.PAR + self.learning_rate * (self.PAR - np.mean(self.fitness))
            self.F = self.F + self.learning_rate * (self.F - np.mean(self.fitness))
            self.CR = self.CR + self.learning_rate * (self.CR - np.mean(self.fitness))
            self.sigma = self.sigma + self.learning_rate * (self.sigma - np.mean(self.fitness))
            if self.fitness[idx] < self.best_fitness:
                self.best_fitness = self.fitness[idx]
                self.best_x = self.x[idx]
        return self.best_x, self.best_fitness