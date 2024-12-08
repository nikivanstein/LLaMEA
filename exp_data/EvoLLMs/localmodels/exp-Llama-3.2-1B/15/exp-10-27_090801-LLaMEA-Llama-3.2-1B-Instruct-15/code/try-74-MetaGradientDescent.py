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

        # Refine the strategy by changing the number of lines of the selected solution
        self.param_values = self.refine_strategy(self.param_values)

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

    def refine_strategy(self, param_values):
        """
        Refine the strategy by changing the number of lines of the selected solution.

        Args:
            param_values (numpy array): The optimized parameter values.

        Returns:
            numpy array: The refined parameter values.
        """
        # Calculate the average number of lines of the selected solution
        avg_lines = np.mean(np.abs(param_values))

        # Change the number of lines of the selected solution based on the probability
        if np.random.rand() < 0.15:
            # Increase the number of lines by 50%
            return copy.deepcopy(param_values) + 1.5 * param_values
        else:
            # Decrease the number of lines by 50%
            return copy.deepcopy(param_values) - 0.5 * param_values