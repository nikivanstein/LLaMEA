# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
import numpy as np

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

            # Update the mutation rate based on the fitness of the individuals
            self.update MutationRate(func_values, population)

        # Return the optimized parameters and the optimized function value
        return population, func(population)

    def update_MutationRate(self, func_values, population):
        """
        Update the mutation rate based on the fitness of the individuals.

        Parameters:
        func_values (numpy array): The fitness values of the individuals.
        population (numpy array): The individuals in the population.
        """
        # Calculate the mutation rate as the proportion of individuals with high fitness
        mutation_rate = np.mean(func_values > 0.5 * func_values.max())

        # Update the mutation rate
        self.mutation_rate = mutation_rate

        # Clip the mutation rate to ensure it stays within the valid range
        self.mutation_rate = max(0.0, min(1.0, self.mutation_rate))

# One-line description with the main idea:
# Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# 
# This algorithm optimizes the black box function using evolutionary algorithms, incorporating a novel mutation strategy that adapts to the fitness landscape of the individuals.