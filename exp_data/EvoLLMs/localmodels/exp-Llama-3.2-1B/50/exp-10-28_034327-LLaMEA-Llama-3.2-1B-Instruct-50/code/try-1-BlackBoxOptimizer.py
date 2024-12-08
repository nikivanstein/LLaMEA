import numpy as np
from collections import deque

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population = deque(maxlen=self.budget)

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
                self.population.append(point)
                return value
        # If the budget is exceeded, return the best point found so far
        return np.max(self.population[-1])

# Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 