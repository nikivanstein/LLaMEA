import numpy as np
import random
import json

class MetaHeuristic:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the meta-heuristic algorithm.

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
        Optimize the black box function `func` using meta-heuristic.

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

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

    def select_next_individual(self, current_individual, current_fitness):
        """
        Select the next individual based on the current fitness.

        Args:
            current_individual (tuple): The current individual.
            current_fitness (float): The current fitness of the individual.

        Returns:
            tuple: The selected individual.
        """
        # Calculate the probability of selecting each individual based on their fitness
        probabilities = np.array([1 / current_fitness] * self.budget)

        # Select the individual with the highest probability
        selected_individual = np.random.choice(current_individual, p=probabilities)

        # Refine the selected individual based on the current fitness
        selected_individual = self.refine_individual(selected_individual, current_individual, current_fitness)

        return selected_individual

    def refine_individual(self, individual, current_individual, current_fitness):
        """
        Refine the selected individual based on the current fitness.

        Args:
            individual (tuple): The selected individual.
            current_individual (tuple): The current individual.
            current_fitness (float): The current fitness of the individual.

        Returns:
            tuple: The refined individual.
        """
        # Calculate the objective function value of the current individual
        current_value = func(current_individual)

        # Calculate the objective function value of the selected individual
        selected_value = func(individual)

        # If the current individual is better, return it
        if current_value > selected_value:
            return individual

        # Otherwise, return the selected individual
        return selected_individual

# One-line description with the main idea
# MetaHeuristic: An optimization algorithm that uses meta-heuristics to search for the optimal solution in the black box function space.

# Code: