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
            new_individuals = []
            for individual in self.x:
                # Randomly choose between PSO and DE operators
                if random.random() < self.random_prob:
                    # PSO operator
                    v = self.w * np.random.uniform(0, 1, (1, self.dim)) + self.c1 * np.abs(individual - self.best_x[:, np.newaxis]) + self.c2 * np.abs(individual - np.mean(individual, axis=0)[:, np.newaxis]) ** self.f
                    new_individual = individual + v
                else:
                    # DE operator
                    v = self.w * np.random.uniform(0, 1, (1, self.dim)) + self.c1 * np.abs(individual - self.best_x[:, np.newaxis]) + self.c2 * np.abs(individual - np.mean(individual, axis=0)[:, np.newaxis]) ** self.f
                    new_individual = individual + v
                # Limit the search space
                new_individual = np.clip(new_individual, self.lower_bound, self.upper_bound)
                new_individuals.append(new_individual)
            self.x = np.array(new_individuals)

            # Evaluate the function at the updated population
            fval = func(self.x)

            # Update the best solution
            if fval < self.fval:
                self.fval = fval
                self.best_x = self.x[np.argmin(fval)]

            # Select the best individual
            self.x = self.x[np.argmin(fval)]

        return self.fval, self.best_x