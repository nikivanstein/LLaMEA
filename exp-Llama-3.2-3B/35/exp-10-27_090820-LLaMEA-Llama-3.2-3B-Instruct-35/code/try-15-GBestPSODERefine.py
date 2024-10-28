import numpy as np
import random

class GBestPSODERefine:
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
        self.refine_prob = 0.35
        self.x = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fval = np.inf
        self.best_x = np.inf

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

            # Refine the strategy with probability 0.35
            if random.random() < self.refine_prob:
                # Randomly select an individual to refine
                idx = np.random.choice(self.population_size, 1, replace=False)
                new_individual = self.x[idx]
                # Refine the individual by adding a random perturbation
                new_individual += np.random.uniform(-0.1, 0.1, self.dim)
                # Ensure the new individual is within the bounds
                new_individual = np.clip(new_individual, self.lower_bound, self.upper_bound)
                # Replace the original individual with the refined one
                self.x[idx] = new_individual

        return self.fval, self.best_x