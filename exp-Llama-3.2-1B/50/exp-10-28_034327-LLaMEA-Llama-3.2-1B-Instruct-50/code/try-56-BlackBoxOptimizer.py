import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.x = None
        self.y = None

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random point in the search space
            self.x = np.random.uniform(-5.0, 5.0, self.dim)
            self.y = func(self.x)
            # Evaluate the function at the point
            self.y = np.max(func(self.x))
            # Check if the point is within the bounds
            if -5.0 <= self.x[0] <= 5.0 and -5.0 <= self.x[1] <= 5.0:
                # If the point is within bounds, update the function value
                self.y = np.max(func(self.x))
                self.func_evals += 1
                return self.y
        # If the budget is exceeded, return the best point found so far
        return np.max(func(self.x))

# Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 