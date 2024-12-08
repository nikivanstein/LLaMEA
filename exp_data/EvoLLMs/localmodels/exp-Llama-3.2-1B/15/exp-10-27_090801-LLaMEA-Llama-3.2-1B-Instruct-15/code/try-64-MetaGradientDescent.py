# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
import random
import copy

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
        self.param_values = self.refine_strategy(self.param_values, func_value)

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

    def refine_strategy(self, param_values, func_value):
        """
        Refine the strategy based on the accumulated noise.

        Args:
            param_values (numpy.ndarray): The current parameter values.
            func_value (float): The objective function value.

        Returns:
            numpy.ndarray: The refined parameter values.
        """
        # Calculate the accumulated noise
        accumulated_noise = self.noise * np.random.normal(0, 1, self.dim)

        # Update the parameter values based on the accumulated noise
        refined_param_values = param_values + accumulated_noise

        # Normalize the parameter values to ensure they are within the search space
        refined_param_values = (refined_param_values + 5.0) / 10.0

        return refined_param_values

class MetaMultiGradientDescent:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the meta-multi-gradient descent algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level

    def __call__(self, func):
        """
        Optimize the black box function `func` using meta-multi-gradient descent.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the parameter values to random values within the search space
        param_values = np.random.uniform(-5.0, 5.0, self.dim)

        # Accumulate noise in the objective function evaluations
        for _ in range(self.budget):
            # Evaluate the objective function with accumulated noise
            func_value = func(param_values)

            # Update the parameter values based on the accumulated noise
            param_values += self.noise_level * np.random.normal(0, 1, self.dim)

        # Refine the strategy based on the accumulated noise
        param_values = self.refine_strategy(param_values, func_value)

        # Return the optimized parameter values and the objective function value
        return param_values, func_value

    def refine_strategy(self, param_values, func_value):
        """
        Refine the strategy based on the accumulated noise.

        Args:
            param_values (numpy.ndarray): The current parameter values.
            func_value (float): The objective function value.

        Returns:
            numpy.ndarray: The refined parameter values.
        """
        # Calculate the accumulated noise
        accumulated_noise = self.noise_level * np.random.normal(0, 1, self.dim)

        # Update the parameter values based on the accumulated noise
        refined_param_values = param_values + accumulated_noise

        # Normalize the parameter values to ensure they are within the search space
        refined_param_values = (refined_param_values + 5.0) / 10.0

        return refined_param_values

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
# MetaGradientDescent: Novel meta-gradient descent algorithm for black box optimization
# MetaMultiGradientDescent: Novel meta-multi-gradient descent algorithm for black box optimization
# ```