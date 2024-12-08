import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population_size = 100
        self.mutation_rate = 0.01

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random point in the search space
            points = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at each point
            values = [func(point) for point in points]
            # Check if the point is within the bounds
            if all(-5.0 <= point <= 5.0 for point in points):
                # If the point is within bounds, update the function value
                self.func_evals += 1
                return values
        # If the budget is exceeded, return the best point found so far
        return np.max(values)

# Iterated Permutation and Cooling Algorithm
# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 