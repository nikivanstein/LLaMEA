# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
import numpy as np
import random
import copy

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

        # Select the fittest individual based on the function values
        fittest_individual = np.argmax(func_values)

        # Create a new population by combining the fittest individual with a mutation strategy
        new_population = []
        for _ in range(population_size):
            # Randomly select a mutation operator
            if random.random() < 0.5:
                # Apply a mutation to the individual
                new_individual = copy.deepcopy(population[fittest_individual])
                new_individual[fittest_individual] += random.uniform(-1, 1)
            else:
                # Keep the individual as it is
                new_individual = copy.deepcopy(population[fittest_individual])

            # Add the new individual to the new population
            new_population.append(new_individual)

        # Replace the old population with the new population
        population = new_population

        # Evaluate the function for each individual in the new population
        func_values = []
        for individual in population:
            func_values.append(func(individual))

        # Select the fittest individual based on the function values
        fittest_individual = np.argmax(func_values)

        # Return the optimized parameters and the optimized function value
        return population[fittest_individual], func(population[fittest_individual])

# One-line description with the main idea
# Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# This algorithm optimizes black box functions using a population-based approach with a mutation strategy.
# The mutation strategy involves randomly changing individual parameters with a probability of 0.5.