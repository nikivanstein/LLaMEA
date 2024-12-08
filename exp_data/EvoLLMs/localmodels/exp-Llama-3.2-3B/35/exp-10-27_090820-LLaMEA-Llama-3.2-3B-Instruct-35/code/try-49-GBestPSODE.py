import numpy as np
import random

class GBestPSODE:
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
        self.mut_prob = 0.35

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
            new_x = self.x[np.argsort(np.abs(self.x - self.best_x))]
            new_x = new_x[:self.population_size]
            for i in range(self.population_size):
                # Select a random individual to refine its strategy
                if random.random() < self.mut_prob:
                    j = random.randint(0, self.population_size - 1)
                    new_x[i] = self.x[j] + np.random.uniform(-0.1, 0.1, self.dim)
                # Apply PSO and DE operators
                v = self.w * np.random.uniform(0, 1, (self.population_size, self.dim)) + self.c1 * np.abs(new_x - self.best_x[:, np.newaxis]) + self.c2 * np.abs(new_x - np.mean(new_x, axis=0)[:, np.newaxis]) ** self.f
                new_x = new_x + v
                # Limit the search space
                new_x = np.clip(new_x, self.lower_bound, self.upper_bound)
                # Evaluate the function at the updated population
                fval = func(new_x)
                # Update the best solution
                if fval < self.fval:
                    self.fval = fval
                    self.best_x = new_x[np.argmin(fval)]
                # Select the best individual
                self.x = new_x[np.argmin(fval)]

        return self.fval, self.best_x