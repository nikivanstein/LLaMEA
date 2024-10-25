# Description: Novel Metaheuristic Algorithm for Black Box Optimization on BBOB Test Suite
# Code: 
import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a given budget and dimensionality.

        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        """
        Optimize a black box function using the BlackBoxOptimizer.

        Args:
        func (function): The black box function to optimize.

        Returns:
        float: The optimized value of the function.
        """
        # Create a copy of the search space to avoid modifying the original
        search_space = self.search_space.copy()

        # Perform the specified number of function evaluations
        for _ in range(self.budget):
            # Evaluate the function at the current search space point
            func_value = func(search_space)

            # Update the search space point to be the next possible point
            # using a probabilistic strategy to avoid overfitting
            if random.random() < 0.5:
                # Use the current point as the new point with a higher probability
                new_point = np.array([self.search_space[:, 0] + np.random.uniform(-1, 1, size=dim),
                                      self.search_space[:, 1] + np.random.uniform(-1, 1, size=dim),
                                      self.search_space[:, 2] + np.random.uniform(-1, 1, size=dim),
                                      self.search_space[:, 3] + np.random.uniform(-1, 1, size=dim)])

                # Normalize the new point to have a length of 1
                new_point = new_point / np.linalg.norm(new_point)
                search_space = np.array([new_point[:, 0], new_point[:, 1], new_point[:, 2], new_point[:, 3]])

            # Update the search space point with a uniform probability
            else:
                # Update the search space point with a uniform probability
                search_space = np.array([self.search_space[:, 0] + np.random.uniform(-1, 1, size=dim),
                                        self.search_space[:, 1] + np.random.uniform(-1, 1, size=dim),
                                        self.search_space[:, 2] + np.random.uniform(-1, 1, size=dim),
                                        self.search_space[:, 3] + np.random.uniform(-1, 1, size=dim)])

        # Return the optimized function value
        return func_value