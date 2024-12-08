import numpy as np
import random

class AdaptiveEvolutionStrategy:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the adaptive evolution strategy algorithm.

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
        Optimize the black box function `func` using adaptive evolution strategy.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the parameter values to random values within the search space
        self.param_values = np.random.uniform(-5.0, 5.0, self.dim)

        # Accumulate noise in the objective function evaluations
        for _ in range(self.budget):
            # Evaluate the objective function with accumulated noise
            func_value = func(self.param_values + self.noise * np.random.normal(0, 1, self.dim))

            # Update the parameter values based on the accumulated noise
            self.param_values += self.noise * np.random.normal(0, 1, self.dim)

            # Apply crossover and mutation to the parameter values
            self.param_values = self.crossover(self.param_values, self.mutation(self.param_values, self.noise_level))

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

    def crossover(self, individual1, individual2):
        """
        Perform crossover between two individuals.

        Args:
            individual1 (np.ndarray): The first individual.
            individual2 (np.ndarray): The second individual.

        Returns:
            np.ndarray: The resulting individual after crossover.
        """
        # Calculate the crossover point
        crossover_point = np.random.randint(0, self.dim)

        # Perform crossover
        child = individual1[:crossover_point] + individual2[crossover_point:]

        return child

    def mutation(self, individual, noise_level):
        """
        Perform mutation on an individual.

        Args:
            individual (np.ndarray): The individual to mutate.
            noise_level (float): The level of noise accumulation.

        Returns:
            np.ndarray: The mutated individual.
        """
        # Calculate the mutation point
        mutation_point = np.random.randint(0, self.dim)

        # Perform mutation
        mutated_individual = individual + noise_level * np.random.normal(0, 1, self.dim)

        return mutated_individual

# Description: Adaptive Evolution Strategy with Crossover and Mutation
# Code: 