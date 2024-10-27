import numpy as np
import random

class GBestPSODEC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.w = 0.7298
        self.c1 = 1.49618
        self.c2 = 2.049912
        self.f = 0.5
        self.x = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fval = np.inf
        self.best_x = np.inf
        self.cma_es = CMAES(self.population_size, self.dim, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the function at the current population
            fval = func(self.x)

            # Update the best solution
            if fval < self.fval:
                self.fval = fval
                self.best_x = self.x[np.argmin(fval)]
                self.x = np.array([self.best_x])

            # Update the population using PSO, DE, and CMA-ES
            self.x = np.vstack((self.x, self.cma_es.optimize(func, self.x)))

            # Limit the search space
            self.x = np.clip(self.x, self.lower_bound, self.upper_bound)

            # Evaluate the function at the updated population
            fval = func(self.x)

            # Update the best solution
            if fval < self.fval:
                self.fval = fval
                self.best_x = self.x[np.argmin(fval)]

            # Select the best individual
            self.x = self.x[np.argmin(fval)]

        return self.fval, self.best_x

class CMAES:
    def __init__(self, population_size, dim, lower_bound, upper_bound):
        self.population_size = population_size
        self.dim = dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.x = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fval = np.inf
        self.best_x = np.inf
        self.mu = 0.2
        self.lambd = 1.0
        self.sigma = 1.0
        self.c1 = 1.49618
        self.c2 = 2.049912
        self.f = 0.5

    def optimize(self, func, x):
        for _ in range(50):
            # Evaluate the function at the current population
            fval = func(x)

            # Update the best solution
            if fval < self.fval:
                self.fval = fval
                self.best_x = x[np.argmin(fval)]

            # Update the population using CMA-ES
            self.x = self.x + self.mu * np.random.normal(0, self.sigma, (self.population_size, self.dim))
            self.x = np.clip(self.x, self.lower_bound, self.upper_bound)
            self.x = self.x[np.argsort(np.abs(self.x - self.best_x))]
            self.x = self.x[:self.population_size]

            # Apply CMA-ES operators
            v = self.c1 * np.abs(self.x - self.best_x[:, np.newaxis]) + self.c2 * np.abs(self.x - np.mean(self.x, axis=0)[:, np.newaxis]) ** self.f
            self.x = self.x + v

        return self.x