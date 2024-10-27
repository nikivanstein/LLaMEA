import numpy as np
import random

class MetaMosaic:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the MetaMosaic algorithm.

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
        Optimize the black box function `func` using MetaMosaic.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the parameter values to random values within the search space
        self.param_values = np.random.uniform(-5.0, 5.0, self.dim)

        # Create a mosaic with the number of tiles equal to the number of function evaluations
        mosaic = np.tile(self.param_values, (self.budget, 1))

        # Accumulate noise in the objective function evaluations
        for _ in range(self.budget):
            # Evaluate the objective function with accumulated noise
            func_value = func(mosaic + self.noise * np.random.normal(0, 1, self.dim))

            # Update the mosaic based on the performance of each tile
            mosaic += self.noise * np.random.normal(0, 1, self.dim)

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

# Example usage:
meta_mosaic = MetaMosaic(budget=100, dim=5)
func = lambda x: x**2  # Example black box function
optimized_values, objective_value = meta_mosaic(func)
print("Optimized values:", optimized_values)
print("Objective value:", objective_value)