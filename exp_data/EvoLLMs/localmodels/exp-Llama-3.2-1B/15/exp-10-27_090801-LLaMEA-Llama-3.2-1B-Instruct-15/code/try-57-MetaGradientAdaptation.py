import numpy as np
import random

class MetaGradientAdaptation:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the meta-gradient adaptation algorithm.

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
        Optimize the black box function `func` using meta-gradient adaptation.

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

            # Update the objective function value based on the performance of the current solution
            self.param_values = self.param_values + self.noise * np.random.normal(0, 1, self.dim) * func_value

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

# Example usage:
meta_gradient_adaptation = MetaGradientAdaptation(100, 10)
optimized_individual, optimized_function_value = meta_gradient_adaptation(func, 10)
print(f"Optimized individual: {optimized_individual}")
print(f"Optimized function value: {optimized_function_value}")