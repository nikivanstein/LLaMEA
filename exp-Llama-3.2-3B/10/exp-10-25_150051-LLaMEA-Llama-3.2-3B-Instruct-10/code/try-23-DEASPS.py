import numpy as np
import random
from scipy.optimize import minimize

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

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        for i in range(self.budget):
            fitness = np.array([func(x) for x in population])
            best_individual = population[np.argmin(fitness)]
            new_population = population + self.adaptive_step_size * np.random.uniform(-1, 1, (self.population_size, self.dim))
            if np.random.rand() < self.adaptive_population_size:
                new_population = new_population[:int(self.population_size * self.adaptive_population_size)]
            if np.random.rand() < self.line_search_probability:
                new_individual = self.line_search(func, new_population, best_individual)
                population[np.argsort(fitness)] = np.concatenate((population[np.argsort(fitness)], new_population[np.argsort(fitness)]))
            else:
                population[np.argsort(fitness)] = np.concatenate((population[np.argsort(fitness)], new_population[np.argsort(fitness)]))
        return population[np.argmin(fitness)]

    def line_search(self, func, new_population, best_individual):
        initial_guess = best_individual
        initial_fitness = func(initial_guess)
        best_guess = initial_guess
        best_fitness = initial_fitness
        for _ in range(10):
            guess = initial_guess + 0.1 * np.random.uniform(-1, 1, self.dim)
            fitness = func(guess)
            if fitness < best_fitness:
                best_guess = guess
                best_fitness = fitness
        return best_guess

# Example usage:
def func(x):
    return np.sum(x**2)

de = DEASPS(budget=100, dim=10)
best_x = de(func)
print("Best x:", best_x)
print("Best f(x):", func(best_x))