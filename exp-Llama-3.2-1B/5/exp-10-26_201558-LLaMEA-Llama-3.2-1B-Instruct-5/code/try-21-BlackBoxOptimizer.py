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
        self.population_size = 100

    def __call__(self, func):
        """
        Optimize the black box function `func` using the given budget and search space.

        Parameters:
        func (function): The black box function to optimize.

        Returns:
        tuple: The optimized parameters and the optimized function value.
        """
        # Initialize the population with random parameters
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        # Evaluate the function for each individual in the population
        for _ in range(self.budget):
            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(np.abs(func(population)))[:, ::-1][:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Apply mutation to each individual
            for i in range(self.population_size):
                if random.random() < 0.05:
                    new_population[i] += random.uniform(-1.0, 1.0)

            # Replace the old population with the new population
            population = new_population

        # Return the optimized parameters and the optimized function value
        return population, func(population)

# Example usage:
def f(x):
    return x[0]**2 + x[1]**2

optimizer = BlackBoxOptimizer(100, 2)
optimized_params, optimized_func = optimizer(__call__(f))

# Print the optimized parameters and function value
print("Optimized parameters:", optimized_params)
print("Optimized function value:", optimized_func(f(optimized_params)))