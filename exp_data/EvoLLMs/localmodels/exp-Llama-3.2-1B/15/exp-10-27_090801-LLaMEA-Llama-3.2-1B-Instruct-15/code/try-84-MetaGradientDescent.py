import numpy as np
import random
import os

class MetaGradientDescent:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the meta-gradient descent algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0
        self.search_space = (-5.0, 5.0, -5.0, 5.0)

    def __call__(self, func):
        """
        Optimize the black box function `func` using meta-gradient descent.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the parameter values to random values within the search space
        self.param_values = np.random.uniform(*self.search_space, self.dim)

        # Accumulate noise in the objective function evaluations
        for _ in range(self.budget):
            # Evaluate the objective function with accumulated noise
            func_value = func(self.param_values + self.noise * np.random.normal(0, 1, self.dim))

            # Update the parameter values based on the accumulated noise
            self.param_values += self.noise * np.random.normal(0, 1, self.dim)

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

    def mutate(self):
        """
        Randomly swap two parameter values in the search space.
        """
        i, j = random.sample(range(self.dim), 2)
        self.param_values[i], self.param_values[j] = self.param_values[j], self.param_values[i]

    def __str__(self):
        return f"MetaGradientDescent(budget={self.budget}, dim={self.dim})"

# Description: MetaGradientDescent algorithm
# Code: 