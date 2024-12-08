# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 
import numpy as np
import random
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

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
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

class IteratedPermutationCooling(BlackBoxOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        best_point = None
        best_value = -np.inf
        for _ in range(self.budget):
            # Generate a random point in the search space
            point = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the point
            value = func(point)
            # Check if the point is within the bounds
            if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
                # If the point is within bounds, update the best point and value
                best_point = point
                best_value = value
        # Return the best point found so far
        return best_point

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 
# ```python
import numpy as np

# Select the Iterated Permutation and Cooling Algorithm
algorithm = IteratedPermutationCooling(100, 5)

# Update the BlackBoxOptimizer
updated_optimizer = BlackBoxOptimizer(100, 5)

# Run the optimization
best_point, best_value = updated_optimizer(func)