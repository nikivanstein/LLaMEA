import random
import numpy as np
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            float: The optimized value of the function.
        """
        # Define the objective function to minimize (negative of the function to maximize)
        def neg_func(x):
            return -func(x)

        # Perform the specified number of function evaluations
        res = differential_evolution(neg_func, self.search_space, bounds=[(-5.0, 5.0)] * self.dim, n_iter=self.budget)

        # Return the optimized value
        return -res.fun

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 