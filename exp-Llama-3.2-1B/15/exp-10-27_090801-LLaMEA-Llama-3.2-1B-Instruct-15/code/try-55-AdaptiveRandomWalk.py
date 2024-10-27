import numpy as np
import random

class AdaptiveRandomWalk:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the adaptive random walk algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0
        self.current_fitness = -np.inf

    def __call__(self, func):
        """
        Optimize the black box function `func` using the adaptive random walk algorithm.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the parameter values to random values within the search space
        self.param_values = np.random.uniform(-5.0, 5.0, self.dim)

        # Initialize the accumulated noise to zero
        self.noise = 0

        # Accumulate noise in the objective function evaluations
        for _ in range(self.budget):
            # Evaluate the objective function with accumulated noise
            func_value = func(self.param_values + self.noise * np.random.normal(0, 1, self.dim))

            # Update the parameter values based on the accumulated noise and the current fitness value
            self.param_values += self.noise * np.random.normal(0, 1, self.dim)
            self.noise += (func_value - self.current_fitness) / self.current_fitness

            # Update the current fitness value
            self.current_fitness = func_value

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

# Code: 
# ```python
# Adaptive Random Walk
# ```
# ```python
# ```python
# ```python
# # Initialize the adaptive random walk algorithm
adaptive_random_walk = AdaptiveRandomWalk(budget=100, dim=10)

# Optimize the black box function
optimized_param_values, optimized_func_value = adaptive_random_walk(func)

# Print the result
print(f"Optimized Parameter Values: {optimized_param_values}")
print(f"Optimized Function Value: {optimized_func_value}")