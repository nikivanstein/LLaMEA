import numpy as np
import random

class DEASPS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.adaptive_step_size = 0.5
        self.adaptive_population_size = 0.5
        self.adaptive_probability = 0.1
        self.lower_bound = -5.0
        self.upper_bound = 5.0

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

            # Refine strategy with 10% probability
            if np.random.rand() < self.adaptive_probability:
                for j in range(self.population_size):
                    new_individual = population[j] + np.random.uniform(-0.1, 0.1, self.dim)
                    population[j] = new_individual

        # Return best individual
        return population[np.argmin(fitness)]

# Example usage:
def func(x):
    return np.sum(x**2)

de = DEASPS(budget=100, dim=10)
best_x = de(func)
print("Best x:", best_x)
print("Best f(x):", func(best_x))