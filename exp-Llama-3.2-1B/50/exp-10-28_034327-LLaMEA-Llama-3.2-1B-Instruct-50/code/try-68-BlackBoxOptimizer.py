import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.search_space = np.linspace(-5.0, 5.0, self.dim)
        self.iterations = 0
        self.cooling_rate = 0.95
        self.iterated_permutations = 0

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random permutation of the current search space
            permutation = np.random.permutation(self.search_space)
            # Evaluate the function at the current permutation
            value = func(permutation)
            # Check if the permutation is within the bounds
            if -5.0 <= permutation[0] <= 5.0 and -5.0 <= permutation[1] <= 5.0:
                # If the permutation is within bounds, update the function value
                self.func_evals += 1
                return value
            # Refine the permutation by iterated permutation
            self.iterated_permutations += 1
            # Generate a new permutation by iterated permutation
            new_permutation = np.random.permutation(self.search_space)
            # Evaluate the function at the new permutation
            value = func(new_permutation)
            # Check if the new permutation is within the bounds
            if -5.0 <= new_permutation[0] <= 5.0 and -5.0 <= new_permutation[1] <= 5.0:
                # If the new permutation is within bounds, update the function value
                self.func_evals += 1
                return value
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 