import numpy as np
import random

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
        for _ in range(self.budget):
            # Select parents using tournament selection
            parents = np.array([self.population[i] for i in fittest_indices])

            # Perform mutation
            mutated_parents = parents.copy()
            for _ in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    mutated_parents[_] += np.random.normal(0, 1, self.dim)

            # Select offspring using tournament selection
            offspring = np.array([self.population[i] for i in np.argsort(mutated_parents)[::-1][:self.population_size]])

            # Replace the old population with the new one
            self.population = np.concatenate((self.population, mutated_parents), axis=0)
            self.population = np.concatenate((self.population, offspring), axis=0)

        # Evaluate the function with the final population
        func_values = np.array([func(x) for x in self.population])
        return func_values

class EvoDiffEvolutionaryDifferentialEvolution:
    def __init__(self, budget, dim, mutation_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.population = EvoDiff(100, dim)

    def __call__(self, func):
        # Initialize the population
        self.population = EvoDiffEvolutionaryDifferentialEvolution(self.budget, dim)

        # Run the evolutionary differential evolution algorithm
        while self.population.population.size < self.budget:
            # Select parents using tournament selection
            parents = np.array([self.population.population[i] for i in np.argsort(self.population.func_values)[::-1][:self.population.population.size]])

            # Perform mutation
            mutated_parents = parents.copy()
            for _ in range(self.population.population.size):
                if np.random.rand() < self.mutation_rate:
                    mutated_parents[_] += np.random.normal(0, 1, self.dim)

            # Select offspring using tournament selection
            offspring = np.array([self.population.population[i] for i in np.argsort(mutated_parents)[::-1][:self.population.population.size]])

            # Replace the old population with the new one
            self.population.population = np.concatenate((self.population.population, mutated_parents), axis=0)
            self.population.population = np.concatenate((self.population.population, offspring), axis=0)

        # Evaluate the function with the final population
        func_values = np.array([func(x) for x in self.population.population])
        return func_values

# One-line description with the main idea
# Evolutionary Differential Evolution Algorithm for Black Box Optimization
# This algorithm uses evolutionary differential evolution to optimize black box functions