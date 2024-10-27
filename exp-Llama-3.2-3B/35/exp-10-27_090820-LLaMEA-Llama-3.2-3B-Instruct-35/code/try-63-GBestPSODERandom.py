import numpy as np
import random

class GBestPSODERandom:
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
        self.random_prob = 0.35

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the function at the current population
            fval = func(self.x)

            # Update the best solution
            if fval < self.fval:
                self.fval = fval
                self.best_x = self.x[np.argmin(fval)]
                self.x = np.array([self.best_x])

            # Update the population using PSO and DE
            new_x = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                if random.random() < self.random_prob:
                    new_x[i] = random.uniform(self.lower_bound, self.upper_bound)
                else:
                    v = self.w * np.random.uniform(0, 1, (self.dim,))
                    new_x[i] = self.x[i] + v + self.c1 * np.abs(self.x[i] - self.best_x[:, np.newaxis]) + self.c2 * np.abs(self.x[i] - np.mean(self.x, axis=0)[:, np.newaxis]) ** self.f
                    new_x[i] = np.clip(new_x[i], self.lower_bound, self.upper_bound)
            self.x = np.vstack((self.x, new_x))

            # Evaluate the function at the updated population
            fval = func(self.x)

            # Update the best solution
            if fval < self.fval:
                self.fval = fval
                self.best_x = self.x[np.argmin(fval)]

            # Select the best individual
            self.x = self.x[np.argmin(fval)]

        return self.fval, self.best_x