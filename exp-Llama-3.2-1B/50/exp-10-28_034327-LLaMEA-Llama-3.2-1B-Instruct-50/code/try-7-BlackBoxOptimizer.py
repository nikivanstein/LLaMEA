import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.current_individual = None

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random point in the search space
            self.current_individual = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the point
            value = func(self.current_individual)
            # Check if the point is within the bounds
            if -5.0 <= self.current_individual[0] <= 5.0 and -5.0 <= self.current_individual[1] <= 5.0:
                # If the point is within bounds, update the function value
                self.func_evals += 1
                return value
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

# Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 