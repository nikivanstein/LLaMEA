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

            # Apply mutation to each individual in the population
            for i in range(population_size):
                if random.random() < 0.05:
                    # Randomly select an individual from the population
                    individual = population[i]

                    # Randomly select a random value from the search space
                    mutation_value = random.uniform(-5.0, 5.0)

                    # Apply mutation to the individual
                    individual[population_size - 1] += mutation_value

                    # Ensure the individual remains within the search space
                    individual[population_size - 1] = max(-5.0, min(individual[population_size - 1], 5.0))

        # Return the optimized parameters and the optimized function value
        return population, func(population)

# One-line description with the main idea
# Novel Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy