import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initializes the BlackBoxOptimizer with a given budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        """
        Evaluates the given black box function using the given budget and returns the optimized result.

        Args:
            func (function): The black box function to optimize.

        Returns:
            float: The optimized result of the function.
        """
        # Limit the number of function evaluations based on the given budget
        evaluations = min(self.budget, func.__code__.co_argcount)

        # Initialize the current point in the search space
        current_point = self.search_space.copy()

        # Perform the specified number of function evaluations
        for _ in range(evaluations):
            # Generate a new point in the search space
            new_point = current_point + np.random.uniform(-1.0, 1.0, self.dim)

            # Evaluate the function at the new point
            func_value = func(new_point)

            # If the function value is better than the current best, update the current point
            if func_value < self.search_space.max():
                current_point = new_point

        # Return the optimized result
        return current_point

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 