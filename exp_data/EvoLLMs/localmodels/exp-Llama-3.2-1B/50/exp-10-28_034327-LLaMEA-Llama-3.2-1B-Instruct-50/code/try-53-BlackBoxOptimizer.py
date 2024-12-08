import numpy as np
import random

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

    def __iter__(self):
        return self

    def __next__(self):
        # Refine the strategy by changing the individual lines of the selected solution
        if random.random() < 0.45:
            # Change the individual lines of the selected solution
            self.func_evals = np.random.randint(1, self.budget + 1)
            # Update the best point found so far
            best_point = np.random.uniform(-5.0, 5.0, self.dim)
            best_value = np.max(func(best_point))
            return (best_point, best_value)
        else:
            # Return the best point found so far
            return (np.random.uniform(-5.0, 5.0, self.dim), np.max(func(np.random.uniform(-5.0, 5.0, self.dim))))

# Description: Iterated Permutation and Cooling Strategy
# Code: 
# ```python
# BlackBoxOptimizer: Iterated Permutation and Cooling Strategy
# Code: 