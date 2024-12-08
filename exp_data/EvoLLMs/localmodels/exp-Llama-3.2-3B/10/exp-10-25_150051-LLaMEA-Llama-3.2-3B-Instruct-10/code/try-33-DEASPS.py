import numpy as np
import random

class DEASPS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.adaptive_step_size = 0.5
        self.adaptive_population_size = 0.5
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.line_search_prob = 0.1

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

        for i in range(self.budget):
            fitness = np.array([func(x) for x in population])

            best_individual = population[np.argmin(fitness)]

            new_population = population + self.adaptive_step_size * np.random.uniform(-1, 1, (self.population_size, self.dim))

            if np.random.rand() < self.adaptive_population_size:
                new_population = new_population[:int(self.population_size * self.adaptive_population_size)]

            line_search_prob = self.line_search_prob if np.random.rand() < self.line_search_prob else 0

            for j in range(self.population_size):
                if line_search_prob > 0:
                    # Perform line search to find the optimal step size
                    step_size = self.line_search(func, population[j], self.lower_bound, self.upper_bound)
                    new_population[j] = population[j] + step_size

            population[np.argsort(fitness)] = np.concatenate((population[np.argsort(fitness)], new_population[np.argsort(fitness)]))

        return population[np.argmin(fitness)]

    def line_search(self, func, x, lower_bound, upper_bound):
        # Perform line search using the Armijo rule
        step_size = 1
        while True:
            new_x = x + step_size * (upper_bound - lower_bound) / 10
            if func(new_x) <= func(x) * (1 - 0.1):
                return new_x
            step_size *= 0.9

# Example usage:
def func(x):
    return np.sum(x**2)

de = DEASPS(budget=100, dim=10)
best_x = de(func)
print("Best x:", best_x)
print("Best f(x):", func(best_x))