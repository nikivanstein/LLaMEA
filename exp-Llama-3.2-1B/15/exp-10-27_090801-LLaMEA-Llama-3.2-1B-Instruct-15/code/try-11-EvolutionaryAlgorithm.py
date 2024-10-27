import numpy as np
import random
import os

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the evolutionary algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0

    def __call__(self, func):
        """
        Optimize the black box function `func` using evolutionary algorithm.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the population with random parameter values
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]

        # Accumulate noise in the objective function evaluations
        for _ in range(self.budget):
            # Evaluate the objective function with accumulated noise
            func_value = func(population[-1] + self.noise * np.random.normal(0, 1, self.dim))

            # Update the parameter values based on the accumulated noise
            population[-1] += self.noise * np.random.normal(0, 1, self.dim)

        # Select the fittest individual
        fittest_individual = population[np.argmax([func(individual) for individual in population])]

        # Refine the solution based on the probability 0.15
        if random.random() < 0.15:
            fittest_individual += np.random.normal(0, 1, self.dim)

        # Return the optimized parameter values and the objective function value
        return fittest_individual, func(fittest_individual)

# One-line description with the main idea
# Evolutionary Algorithm for Black Box Optimization
# Refine the solution based on the probability 0.15