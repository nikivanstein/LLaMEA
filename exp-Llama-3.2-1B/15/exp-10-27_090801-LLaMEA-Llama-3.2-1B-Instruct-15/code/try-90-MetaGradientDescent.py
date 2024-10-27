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

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

    def update(self, func, budget):
        """
        Update the meta-gradient descent algorithm based on the cumulative objective function evaluations.

        Args:
            func (callable): The black box function to optimize.
            budget (int): The number of function evaluations to consider.

        Returns:
            tuple: A tuple containing the optimized parameter values and the updated objective function value.
        """
        # Initialize the parameter values to random values within the search space
        self.param_values = np.random.uniform(-5.0, 5.0, self.dim)

        # Accumulate noise in the objective function evaluations
        for _ in range(budget):
            # Evaluate the objective function with accumulated noise
            func_value = func(self.param_values + self.noise * np.random.normal(0, 1, self.dim))

            # Update the parameter values based on the accumulated noise
            self.param_values += self.noise * np.random.normal(0, 1, self.dim)

        # Return the optimized parameter values and the updated objective function value
        return self.param_values, func_value

# Example usage:
def func1(x):
    return x[0]**2 + x[1]**2

def func2(x):
    return x[0]**2 + x[1]**2 + x[2]**2

algorithm = MetaGradientDescent(budget=100, dim=2)
optimized_params1, func_value1 = algorithm(func1)
optimized_params2, func_value2 = algorithm(func2)

# Update the algorithm based on the cumulative objective function evaluations
budget = 50
new_params, new_func_value = algorithm.update(func1, budget)
optimized_params3, new_func_value = algorithm.update(func2, budget)

# Print the results
print("Optimized parameters for func1:", optimized_params1)
print("Objective function value for func1:", new_func_value)
print("Optimized parameters for func2:", optimized_params2)
print("Objective function value for func2:", new_func_value)