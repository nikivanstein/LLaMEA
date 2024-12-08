# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
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

            # Update the population with a novel mutation strategy
            for _ in range(population_size // 2):
                # Select a random individual from the population
                individual = population[np.random.choice(population_size)]

                # Apply a mutation to the individual
                mutated_individual = individual + random.uniform(-1.0, 1.0)

                # Check if the mutated individual is within the search space
                if np.any(mutated_individual < -5.0) or np.any(mutated_individual > 5.0):
                    # If not, reset the mutated individual to the lower bound
                    mutated_individual = np.clip(mutated_individual, -5.0, 5.0)

                # Add the mutated individual to the new population
                population = np.concatenate([population, [mutated_individual]])

        # Return the optimized parameters and the optimized function value
        return population, func(population)

# One-line description with the main idea
# Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Optimizes black box functions using evolutionary algorithms with a novel mutation strategy