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
        self.refine_prob = 0.35

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the function at the current population
            fval = func(self.x)

            # Update the best solution
            if fval < self.fval:
                self.fval = fval
                self.best_x = self.x[np.argmin(fval)]
                self.x = np.array([self.best_x])

            # Refine the current solution with a probability of 0.35
            if random.random() < self.refine_prob:
                # Select a random individual to refine
                idx = np.random.randint(0, self.population_size)
                # Create a new individual by perturbing the selected individual
                new_individual = self.x[idx]
                new_individual += np.random.uniform(-0.1, 0.1, self.dim)
                # Limit the search space
                new_individual = np.clip(new_individual, self.lower_bound, self.upper_bound)
                # Evaluate the function at the new individual
                fval = func(new_individual)
                # Update the best solution if necessary
                if fval < self.fval:
                    self.fval = fval
                    self.best_x = new_individual

            # Update the population using PSO and DE
            self.x = np.vstack((self.x, np.array([random.uniform(self.lower_bound, self.upper_bound) for _ in range(self.dim)])))
            self.x = self.x[np.argsort(np.abs(self.x - self.best_x))]
            self.x = self.x[:self.population_size]

            # Apply PSO and DE operators
            v = self.w * np.random.uniform(0, 1, (self.population_size, self.dim)) + self.c1 * np.abs(self.x - self.best_x[:, np.newaxis]) + self.c2 * np.abs(self.x - np.mean(self.x, axis=0)[:, np.newaxis]) ** self.f
            self.x = self.x + v

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