# Description: Meta-Gradient Descent with Adaptive Learning Rate
# Code: 
# ```python
import numpy as np
import random
import os

class MetaGradientDescent:
    def __init__(self, budget, dim, noise_level=0.1, learning_rate=0.01, alpha=0.1):
        """
        Initialize the meta-gradient descent algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
            learning_rate (float, optional): The learning rate for the meta-gradient descent algorithm. Defaults to 0.01.
            alpha (float, optional): The adaptive learning rate. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.param_values = np.random.uniform(-5.0, 5.0, self.dim)
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
        for _ in range(self.budget):
            # Evaluate the objective function with accumulated noise
            func_value = func(self.param_values + self.noise * np.random.normal(0, 1, self.dim))

            # Update the parameter values based on the accumulated noise
            self.param_values += self.noise * np.random.normal(0, 1, self.dim)

            # Update the noise level for the next iteration
            self.noise = np.random.uniform(-self.noise_level, self.noise_level, self.dim)

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

    def update_learning_rate(self, func_value):
        """
        Update the learning rate based on the objective function value.

        Args:
            func_value (float): The objective function value.
        """
        # Calculate the learning rate
        self.learning_rate *= (1 - self.alpha) / 1000
        # Update the learning rate for the next iteration
        self.alpha = self.alpha / 1000

    def optimize(self, func):
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

            # Update the noise level for the next iteration
            self.noise = np.random.uniform(-self.noise_level, self.noise_level, self.dim)

        # Update the learning rate
        self.update_learning_rate(func_value)

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

# One-line description with the main idea
# Meta-Gradient Descent with Adaptive Learning Rate
# Optimizes the black box function using meta-gradient descent with adaptive learning rate