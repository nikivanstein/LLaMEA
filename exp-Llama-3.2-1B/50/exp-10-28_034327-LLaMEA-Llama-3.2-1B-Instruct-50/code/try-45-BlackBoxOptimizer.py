import numpy as np

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
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

class IteratedPermutationCooling:
    def __init__(self, func, budget, dim):
        self.func = func
        self.budget = budget
        self.dim = dim
        self.iterations = 0
        self.permutations = None

    def __call__(self, func):
        while self.iterations < self.budget:
            # Generate a random permutation of the search space
            permutation = np.random.permutation(self.dim)
            # Evaluate the function at each point in the permutation
            values = [self.func(point) for point in permutation]
            # Check if the point is within the bounds
            if -5.0 <= permutation[0] <= 5.0 and -5.0 <= permutation[1] <= 5.0:
                # If the point is within bounds, update the function value
                self.func_evals += 1
                return np.max(values)
            # Update the permutation
            self.permutations = permutation
            # Cool down the algorithm
            self.iterations += 1
            # Calculate the probability of changing the permutation
            prob = np.random.rand() < 0.45
            # If the probability is less than 0.5, change the permutation
            if prob < 0.5:
                permutation = np.random.permutation(self.dim)
        # If the budget is exceeded, return the best point found so far
        return np.max(self.func(np.random.uniform(-5.0, 5.0, self.dim)))

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 