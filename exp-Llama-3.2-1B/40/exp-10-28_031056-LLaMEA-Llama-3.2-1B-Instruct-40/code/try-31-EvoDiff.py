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
        bounds = [(-5.0, 5.0), (-5.0, 5.0)] * self.dim
        res = differential_evolution(func, bounds, args=[func_values], x0=self.population, maxiter=self.budget)

        # Refine the strategy using tournament selection
        tournament_size = 5
        tournament_indices = np.random.choice(fittest_indices, size=tournament_size, replace=False)
        tournament_values = np.array([func(x) for x in tournament_indices])
        tournament_indices = np.argsort(tournament_values)[::-1][:tournament_size]
        refined_individual = np.array([self.population[i] for i in tournament_indices])
        refined_individual = np.concatenate((refined_individual, self.population[tournament_indices[tournament_size:]]), axis=0)
        refined_individual = np.concatenate((refined_individual, tournament_indices[tournament_size:]), axis=0)

        # Replace the old population with the new one
        self.population = np.concatenate((self.population, refined_individual), axis=0)
        self.population = np.concatenate((self.population, tournament_indices[tournament_size:]), axis=0)

        # Evaluate the function with the final population
        func_values = np.array([func(x) for x in self.population])
        return func_values

# Description: A novel evolutionary differential evolution algorithm that leverages the concept of evolutionary differences to optimize black box functions.
# Code: 