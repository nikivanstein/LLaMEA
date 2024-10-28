import numpy as np
from scipy.optimize import differential_evolution
from collections import deque

class EvoDiff:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.init_population()

    def init_population(self):
        # Initialize the population with random solutions
        return np.random.uniform(-5.0, 5.0, self.dim) + np.random.normal(0, 1, self.dim)

    def __call__(self, func):
        # Evaluate the black box function with the current population
        func_values = np.array([func(x) for x in self.population])

        # Select the fittest solutions
        fittest_indices = np.argsort(func_values)[::-1][:self.population_size]

        # Evolve the population using evolutionary differential evolution
        bounds = [(-5.0, 5.0)] * self.dim
        result = differential_evolution(func, bounds, args=(func_values,), x0=self.population, tol=1e-6, maxiter=self.budget)

        # Refine the strategy by changing the mutation rate
        if result.success:
            new_individual = self.evaluate_fitness(result.x)
            mutation_rate = min(0.4, self.mutation_rate)
            self.population = np.concatenate((self.population, [new_individual] * (self.population_size - len(new_individual))), axis=0)
            self.mutation_rate = mutation_rate
        else:
            self.population = np.concatenate((self.population, result.x), axis=0)

        # Evaluate the function with the final population
        func_values = np.array([func(x) for x in self.population])
        return func_values

    def evaluate_fitness(self, individual):
        func = np.vectorize(individual)(self.dim)
        func_values = np.array([func(x) for x in individual])
        return func_values