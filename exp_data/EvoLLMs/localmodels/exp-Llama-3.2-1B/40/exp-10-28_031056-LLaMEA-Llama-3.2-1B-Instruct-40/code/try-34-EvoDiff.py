import numpy as np
from scipy.optimize import differential_evolution

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
        result = differential_evolution(self.budget, [(x, func(x)) for x in self.population], x0=self.population, bounds=[(-5.0, 5.0), (-np.inf, np.inf)], method="Nelder-Mead", popcount=0.4, tol=1e-6)
        mutated_population = np.array([self.population[i] + np.random.normal(0, 1, self.dim) for i in range(self.population_size)])

        # Select offspring using tournament selection
        offspring = np.array([self.population[i] for i in np.argsort(mutated_population)[::-1][:self.population_size]])

        # Replace the old population with the new one
        self.population = np.concatenate((self.population, mutated_population), axis=0)
        self.population = np.concatenate((self.population, offspring), axis=0)

        # Evaluate the function with the final population
        func_values = np.array([func(x) for x in self.population])
        return func_values