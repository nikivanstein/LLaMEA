import numpy as np
import random

class MetaHeuristicOptimization:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the meta-heuristic optimization algorithm.

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
        Optimize the black box function `func` using meta-heuristic optimization.

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

        # Apply the "cooling" mechanism to prevent rapid convergence
        self.noise *= 0.99
        if self.noise < 0.01:
            self.noise = 0.01

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

# **Example Usage:**
if __name__ == "__main__":
    # Define a sample black box function
    def func(x):
        return x[0]**2 + x[1]**2 + x[2]**2

    # Initialize the meta-heuristic optimization algorithm
    algo = MetaHeuristicOptimization(budget=100, dim=2, noise_level=0.1)

    # Optimize the black box function using the algorithm
    optimized_params, optimized_func_value = algo(func)

    # Print the optimized parameter values and the objective function value
    print("Optimized Parameters:", optimized_params)
    print("Optimized Objective Function Value:", optimized_func_value)