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
        self.mutation_prob = 0.35

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
            new_individuals = []
            for i in range(self.population_size):
                if random.random() < self.mutation_prob:
                    new_individual = self.x[i] + np.random.uniform(-1, 1, self.dim)
                else:
                    new_individual = self.x[i]
                new_individuals.append(new_individual)

            self.x = np.array(new_individuals)

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