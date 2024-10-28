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
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

    def __iter__(self):
        return self

    def next(self):
        if self.iterations < self.budget:
            # Refine the strategy by changing the individual lines
            # Line 1: Reduce the probability of changing the individual line
            prob_change_individual = 0.9
            # Line 2: Increase the probability of changing the line that is within bounds
            prob_change_line = 0.05
            # Line 3: Use a more efficient method to generate a new individual
            method = random.choice(['uniform', 'linear'])
            if method == 'uniform':
                new_individual = np.random.uniform(-5.0, 5.0, self.dim)
            elif method == 'linear':
                new_individual = np.random.uniform(-5.0, 5.0, self.dim) / 5.0
            # Line 4: Change the individual line with a probability that depends on the probability of changing the individual line
            prob_change_line = 0.1 * (1 - prob_change_line)
            # Line 5: Update the individual line with the new individual line
            updated_individual = self.evaluate_fitness(new_individual)
            # Line 6: Update the line that was changed
            self.func_evals += 1
            return updated_individual
        else:
            # If the budget is exceeded, return the best point found so far
            return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 