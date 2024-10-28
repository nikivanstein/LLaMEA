import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.iterations = 0

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the point
            value = func(point)
            # Check if the point is within the bounds
            if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
                # If the point is within bounds, update the function value
                self.func_evals += 1
                return value
            # If the point is outside bounds, use the current best point
            if self.iterations % 10 == 0:
                best_point = np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))
                if value > best_point:
                    point = best_point
                else:
                    point = np.random.uniform(-5.0, 5.0, self.dim)
            # Use the Iterated Permutation and Cooling Algorithm
            if random.random() < 0.5:
                # Select a new point using the Iterated Permutation
                new_point = np.random.permutation(point)
            else:
                # Select a new point using the Cooling Algorithm
                new_point = point - np.random.normal(0, 0.1, self.dim)
            # Check if the new point is within the bounds
            if -5.0 <= new_point[0] <= 5.0 and -5.0 <= new_point[1] <= 5.0:
                # If the new point is within bounds, update the function value
                self.func_evals += 1
                return value
            # If the new point is outside bounds, return the best point found so far
            return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 
# ```python
# Iterated Permutation and Cooling Algorithm
# ```