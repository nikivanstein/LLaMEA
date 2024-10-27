import numpy as np
import random

class MetaMaze:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the MetaMaze algorithm.

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
        Optimize the black box function `func` using MetaMaze.

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

        # Refine the strategy by changing the individual lines of the selected solution
        self.param_values = [self.param_values + self.noise * np.random.normal(0, 1, self.dim) for _ in range(10)]

        # Return the optimized parameter values and the objective function value
        return self.param_values, func(self.param_values)

# Example usage
meta_maze = MetaMaze(100, 10)
func = lambda x: x**2
optimized_values, _ = meta_maze(func)
print(optimized_values)