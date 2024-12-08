import numpy as np
import random
import time

class DEASPS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.adaptive_step_size = 0.5
        self.adaptive_population_size = 0.5
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.line_search_probability = 0.1
        self.line_search_step_size = 0.01
        self.line_search_max_iter = 10

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

        for i in range(self.budget):
            # Calculate fitness values
            fitness = np.array([func(x) for x in population])

            # Select best individual
            best_individual = population[np.argmin(fitness)]

            # Create new population
            new_population = population + self.adaptive_step_size * np.random.uniform(-1, 1, (self.population_size, self.dim))

            # Adaptive population size
            if np.random.rand() < self.adaptive_population_size:
                new_population = new_population[:int(self.population_size * self.adaptive_population_size)]

            # Replace worst individual with best individual
            population[np.argsort(fitness)] = np.concatenate((population[np.argsort(fitness)], new_population[np.argsort(fitness)]))

            # Line search
            if np.random.rand() < self.line_search_probability:
                line_search_start = time.time()
                best_individual = self.line_search(best_individual, func, self.line_search_step_size, self.line_search_max_iter)
                print(f"Line search took {time.time() - line_search_start} seconds")
                if np.isnan(best_individual).any():
                    break

        # Return best individual
        return population[np.argmin(fitness)]

    def line_search(self, individual, func, step_size, max_iter):
        best_individual = individual
        best_fitness = func(best_individual)
        for _ in range(max_iter):
            new_individual = best_individual + step_size * np.sign(np.random.uniform(-1, 1, self.dim))
            new_fitness = func(new_individual)
            if new_fitness < best_fitness:
                best_individual = new_individual
                best_fitness = new_fitness
        return best_individual

# Example usage:
def func(x):
    return np.sum(x**2)

de = DEASPS(budget=100, dim=10)
best_x = de(func)
print("Best x:", best_x)
print("Best f(x):", func(best_x))