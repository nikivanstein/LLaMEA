import numpy as np
import random

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

    def __call__(self, func):
        """
        Optimize the black box function `func` using meta-gradient descent.

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

        # Refine the strategy based on the accumulated noise
        self.param_values = self.refine_strategy(self.param_values)

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

    def refine_strategy(self, param_values):
        """
        Refine the strategy based on the accumulated noise.

        Args:
            param_values (numpy array): The current parameter values.

        Returns:
            numpy array: The refined parameter values.
        """
        # Calculate the accumulated noise value
        noise_value = np.mean(np.abs(self.noise * np.random.normal(0, 1, self.dim)))

        # Refine the parameter values based on the accumulated noise
        param_values = param_values + noise_value * np.random.normal(0, 1, self.dim)

        # Ensure the parameter values stay within the search space
        param_values = np.clip(param_values, -5.0, 5.0)

        return param_values

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 