import numpy as np
import random

class DCHCS_AN:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.alpha = 1.5
        self.beta = 0.5
        self.p_a = 0.5
        self.p_c = 0.5
        self.sigma = 0.1
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
                    if random.random() < self.p_a:
                        r1, r2, r3 = random.sample(range(self.population_size), 3)
                        while r1 == idx or r2 == idx or r3 == idx:
                            r1, r2, r3 = random.sample(range(self.population_size), 3)
                        x_new = self.x[r1] + self.alpha * (self.x[r2] - self.x[r3])
                        x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                        y_new = func(x_new)
                        if y_new < self.fitness[j]:
                            self.x[j] = x_new
                            self.fitness[j] = y_new
                    else:
                        x_new = self.x[j] + self.beta * np.random.normal(0, 1, self.dim)
                        x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                        y_new = func(x_new)
                        if y_new < self.fitness[j]:
                            self.x[j] = x_new
                            self.fitness[j] = y_new
            self.p_a = self.p_a + 0.01 * (self.p_a - self.p_c)
            self.p_c = max(0.1, min(1.0, self.p_a))
            self.sigma = self.sigma + 0.01 * (self.sigma - self.fitness[idx])
            if self.fitness[idx] < self.best_fitness:
                self.best_fitness = self.fitness[idx]
                self.best_x = self.x[idx]
        return self.best_x, self.best_fitness