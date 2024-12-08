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

        # Refine the solution using a new individual
        new_individual = self.evaluate_fitness(self.param_values)
        self.param_values = new_individual

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

    def evaluate_fitness(self, param_values):
        """
        Evaluate the fitness of the given parameter values.

        Args:
            param_values (numpy.ndarray): The parameter values to evaluate.

        Returns:
            float: The fitness value of the given parameter values.
        """
        # Evaluate the black box function at the given parameter values
        func_value = func(param_values)
        # Return the fitness value
        return func_value

# One-Liner Description: 
# MetaHeuristic Algorithm for Black Box Optimization with Refining Strategy
# Code: 
# ```python
# MetaGradientDescent: (MetaGradientDescent, RefineStrategy)
# def __init__(self, budget, dim, noise_level=0.1):
#    ...
# def __call__(self, func):
#    ...
# def evaluate_fitness(self, param_values):
#    ...
# class RefineStrategy:
#     def __init__(self, budget, dim):
#        ...
#     def __call__(self, func, param_values):
#        ...
# ```