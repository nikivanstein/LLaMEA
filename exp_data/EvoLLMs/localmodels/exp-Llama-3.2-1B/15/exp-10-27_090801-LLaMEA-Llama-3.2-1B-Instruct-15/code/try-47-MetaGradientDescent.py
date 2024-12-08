# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
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

    def __str__(self):
        return f"MetaGradientDescent(budget={self.budget}, dim={self.dim})"

# One-line description with main idea
# MetaGradientDescent: Novel metaheuristic algorithm for black box optimization using meta-gradient descent with refinement strategy.

def meta_gradient_descent_refined(budget, dim, noise_level):
    """
    Novel metaheuristic algorithm for black box optimization using meta-gradient descent with refinement strategy.

    Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the problem.
        noise_level (float): The level of noise accumulation.

    Returns:
        tuple: A tuple containing the optimized parameter values and the objective function value.
    """
    meta_gradient_descent = MetaGradientDescent(budget, dim, noise_level)
    # Select the solution with the highest fitness value
    selected_solution = meta_gradient_descent
    # Refine the selected solution based on the probability 0.15
    if random.random() < 0.15:
        selected_solution.param_values = np.random.uniform(-5.0, 5.0, selected_solution.dim)
    return selected_solution.__call__(np.random.normal(0, 1, selected_solution.dim))

# Test the algorithm
selected_solution = meta_gradient_descent_refined(100, 10, 0.1)
print(selected_solution)