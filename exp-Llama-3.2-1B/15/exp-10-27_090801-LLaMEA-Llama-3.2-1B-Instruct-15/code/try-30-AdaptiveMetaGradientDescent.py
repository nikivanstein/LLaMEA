import numpy as np
import random
import os

class AdaptiveMetaGradientDescent:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the adaptive meta-gradient descent algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0
        self.search_spaces = {
            'uniform': (np.linspace(-5.0, 5.0, dim), np.linspace(-5.0, 5.0, dim)),
            'bounded': (np.linspace(-5.0, 5.0, dim), np.linspace(-5.0, 5.0, dim), (5.0, 5.0))
        }
        self.best_individual = None
        self.best_fitness = -np.inf
        self.best_algorithm = None

    def __call__(self, func, algorithm_name='MetaGradientDescent'):
        """
        Optimize the black box function `func` using an adaptive meta-gradient descent algorithm.

        Args:
            func (callable): The black box function to optimize.
            algorithm_name (str, optional): The name of the algorithm. Defaults to 'MetaGradientDescent'.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Evaluate the objective function with accumulated noise
        for _ in range(self.budget):
            # Choose the best search space and algorithm
            search_space, bounds, algorithm = self.search_spaces[algorithm_name]
            noise = self.noise * np.random.normal(0, 1, len(search_space))

            # Evaluate the objective function with accumulated noise
            func_value = func(search_space + noise * np.random.normal(0, 1, len(search_space)))

            # Update the parameter values based on the accumulated noise
            self.param_values = search_space + noise * np.random.normal(0, 1, len(search_space))

            # Evaluate the fitness of the current individual
            fitness = func(self.param_values)

            # Update the best individual and fitness if necessary
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = self.param_values
                self.best_algorithm = algorithm_name

        # Return the optimized parameter values and the objective function value
        return self.best_individual, self.best_fitness

# Description: Adaptive Meta-Gradient Descent for Black-Box Optimization
# Code: 