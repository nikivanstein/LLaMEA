import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a given budget and dimensionality.

        Parameters:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimize the black box function `func` using the given budget and search space.

        Parameters:
        func (function): The black box function to optimize.

        Returns:
        tuple: The optimized parameters and the optimized function value.
        """
        # Initialize the population size
        population_size = 100

        # Initialize the population with random parameters
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))

        # Evaluate the function for each individual in the population
        for _ in range(self.budget):
            # Evaluate the function for each individual in the population
            func_values = func(population)

            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(func_values)[::-1][:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Replace the old population with the new population
            population = new_population

            # Select a new population with a modified fitness function
            def fitness(individual):
                return np.sum(np.abs(individual - func(individual)))

            new_population = self.fitness_to_new_population(population, func, fitness)

        # Return the optimized parameters and the optimized function value
        return population, func(population)

    def fitness_to_new_population(self, population, func, fitness):
        """
        Select a new population with a modified fitness function.

        Parameters:
        population (numpy array): The current population.
        func (function): The black box function to optimize.
        fitness (function): The modified fitness function.

        Returns:
        numpy array: The new population.
        """
        # Create a new population by combining the fittest individuals
        new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

        # Replace the old population with the new population
        population = new_population

        # Calculate the fitness of each individual in the new population
        fitness_values = fitness(population)

        # Select the fittest individuals based on the fitness values
        fittest_individuals = np.argsort(fitness_values)[::-1][:self.population_size // 2]

        # Create a new population by combining the fittest individuals
        new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

        return new_population

# One-line description with the main idea
# Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# 
# This algorithm optimizes black box functions using evolutionary algorithms with a modified fitness function, which combines the strengths of mutation and crossover.