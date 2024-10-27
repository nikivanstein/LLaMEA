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
            new_individuals = np.array([self.x[np.random.choice(self.population_size, 1)[0]] + np.random.uniform(-1, 1, self.dim) for _ in range(self.population_size)])
            new_individuals = np.clip(new_individuals, self.lower_bound, self.upper_bound)

            # Apply PSO and DE operators
            v = self.w * np.random.uniform(0, 1, (self.population_size, self.dim)) + self.c1 * np.abs(new_individuals - self.best_x[:, np.newaxis]) + self.c2 * np.abs(new_individuals - np.mean(new_individuals, axis=0)[:, np.newaxis]) ** self.f
            new_individuals = new_individuals + v

            # Randomly refine the best individual
            if np.random.rand() < self.random_prob:
                best_individual = self.x[np.argmin(fval)]
                new_individuals[np.random.choice(self.population_size, 1)[0]] = best_individual + np.random.uniform(-1, 1, self.dim)

            # Limit the search space
            new_individuals = np.clip(new_individuals, self.lower_bound, self.upper_bound)

            # Evaluate the function at the updated population
            fval = func(new_individuals)

            # Update the best solution
            if fval < self.fval:
                self.fval = fval
                self.best_x = new_individuals[np.argmin(fval)]

            # Select the best individual
            self.x = new_individuals[np.argmin(fval)]

        return self.fval, self.best_x