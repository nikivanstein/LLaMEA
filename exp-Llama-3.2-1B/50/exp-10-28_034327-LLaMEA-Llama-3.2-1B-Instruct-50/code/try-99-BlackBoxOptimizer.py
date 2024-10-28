import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.iterations = 0
        self.cooling_rate = 0.95
        self.iterated_permutations = 0

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

    def iterated_permutation(self, func):
        # Refine the strategy by changing the number of iterations
        self.iterations += 1
        # Generate a random permutation of the search space
        permutation = np.random.permutation(self.dim)
        # Generate a random point in the permutation
        point = np.random.uniform(-5.0, 5.0, self.dim)
        # Evaluate the function at the point
        value = func(point)
        # Check if the point is within the bounds
        if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
            # If the point is within bounds, update the function value
            self.func_evals += 1
            return value
        # If the point is not within bounds, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 
# ```python
def bbo(func, budget, dim):
    optimizer = BlackBoxOptimizer(budget, dim)
    for _ in range(1000):  # Run for 1000 iterations
        point = optimizer.iterated_permutation(func)
        value = optimizer.func(point)
        if value > 0:
            return point
    return None

# Example usage
func = lambda x: np.sin(x)
budget = 100
dim = 2
result = bbo(func, budget, dim)
print(result)