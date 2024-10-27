import numpy as np
import random
from scipy.optimize import differential_evolution

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

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

    def optimize(self, func, initial_param_values, max_iter=100, step_size=0.1):
        """
        Optimize the black box function `func` using the meta-gradient descent algorithm.

        Args:
            func (callable): The black box function to optimize.
            initial_param_values (list): The initial parameter values.
            max_iter (int, optional): The maximum number of iterations. Defaults to 100.
            step_size (float, optional): The step size for updating parameter values. Defaults to 0.1.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the current parameter values
        current_param_values = initial_param_values

        # Iterate until the maximum number of iterations is reached
        for _ in range(max_iter):
            # Evaluate the objective function with accumulated noise
            func_value = func(current_param_values + self.noise * np.random.normal(0, 1, self.dim))

            # Update the parameter values based on the accumulated noise
            current_param_values += self.noise * np.random.normal(0, 1, self.dim)

            # Check if the maximum number of iterations is reached
            if np.abs(func_value - func(current_param_values)) < 1e-6:
                break

        # Return the optimized parameter values and the objective function value
        return current_param_values, func_value