import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer.

        Parameters:
        budget (int): The number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimize the black box function using the given budget.

        Parameters:
        func (function): The black box function to optimize.

        Returns:
        float: The optimized value of the function.
        """
        # Initialize the search space with random values
        x = np.random.uniform(-5.0, 5.0, self.dim)
        
        # Perform the given number of function evaluations
        for _ in range(self.budget):
            # Evaluate the function at the current point
            y = func(x)
            
            # Update the search space if the current function value is better
            if y > x[-1]:
                x = stgd(x, func, 0.1, 0.01)  # Update with adaptive line search
            
            # Update the search space using Stochastic Gradient Descent
            x = stgd(x, func, 0.1, 0.01)  # Update with adaptive line search

        # Return the optimized value of the function
        return x[-1]

# One-line description with the main idea
# "Novel Metaheuristic for Solving Black Box Optimization Problems: Iteratively Refining the Search Space using Stochastic Gradient Descent with Adaptive Line Search"