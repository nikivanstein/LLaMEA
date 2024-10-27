# Description: Adaptive BBOB Metaheuristic: An efficient and adaptive optimization algorithm for solving black box optimization problems.
# Code: 
# ```python
import numpy as np
import random
from scipy.optimize import minimize
from collections import deque

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        """
        Initialize the BBOBMetaheuristic with a given budget and dimensionality.

        Args:
        - budget: The maximum number of function evaluations allowed.
        - dim: The dimensionality of the optimization problem.
        """
        self.budget = budget
        self.dim = dim
        self.func = None
        self.space = None
        self.x = None
        self.f = None
        self.logger = None

    def __call__(self, func):
        """
        Optimize the black box function `func` using `self.budget` function evaluations.

        Args:
        - func: The black box function to be optimized.

        Returns:
        - The optimized function value.
        """
        if self.func is None:
            self.func = func
            self.space = np.random.uniform(-5.0, 5.0, (self.dim,))
            self.x = np.random.uniform(-5.0, 5.0, (self.dim,))
            self.f = self.func(self.x)
        else:
            while self.budget > 0:
                # Sample a new point in the search space
                self.x = np.random.uniform(-5.0, 5.0, (self.dim,))
                # Evaluate the function at the new point
                self.f = self.func(self.x)
                # Check if the new point is better than the current point
                if self.f < self.f + 1e-6:  # add a small value to avoid division by zero
                    # Update the current point
                    self.x = self.x
                    self.f = self.f
                # Update the logger
                self.logger = self.logger or deque(maxlen=100)
                self.logger.append(self.f)

                # Refine the strategy based on the logger
                if self.logger[-1] < 0.2 * self.f:
                    # Increase the budget
                    self.budget += 1
                    # Update the current point
                    self.x = np.random.uniform(-5.0, 5.0, (self.dim,))
                    self.f = self.func(self.x)
                else:
                    # Decrease the budget
                    self.budget -= 1
                    # Update the current point
                    self.x = np.random.uniform(-5.0, 5.0, (self.dim,))
                    self.f = self.func(self.x)

        # Return the optimized function value
        return self.f

# Description: Adaptive BBOB Metaheuristic: An efficient and adaptive optimization algorithm for solving black box optimization problems.
# Code: 
# ```python
bboo_metaheuristic = BBOBMetaheuristic(budget=1000, dim=2)