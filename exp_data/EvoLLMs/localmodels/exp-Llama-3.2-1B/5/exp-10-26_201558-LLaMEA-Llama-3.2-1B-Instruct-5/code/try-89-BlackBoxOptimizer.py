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
            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(func(population))[:population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Apply a novel mutation strategy
            for i in range(new_population.shape[0]):
                if random.random() < 0.05:  # Change this probability to refine the strategy
                    new_population[i] = func(new_population[i])

            # Replace the old population with the new population
            population = new_population

        # Return the optimized parameters and the optimized function value
        return population, func(population)

# One-line description with main idea
# Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Optimizes the black box function using a budget of 100 evaluations, with a population size of 100 and a search space of 5 dimensions
# The mutation strategy involves changing a random individual with a probability of 5%