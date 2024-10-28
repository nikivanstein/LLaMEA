import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

    def __call__(self, func):
        if self.func_values is None:
            # Refine the search space using a variant of the simulated annealing algorithm
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break
            # Use the current function values to guide the next iteration
            self.func_values = np.array([func(self.func_values[i]) for i in range(self.dim)])

        # Select a new point using the current function values
        idx = random.choices(range(self.dim), weights=self.func_values)[0]
        self.func_values[idx] = func(self.func_values[idx])

        # Refine the search space using a variant of the simulated annealing algorithm
        delta = func(self.func_values[idx])
        if np.random.rand() < 0.5:
            # Increase the upper bound
            self.func_values = np.maximum(self.func_values, func(self.func_values[idx] + 1.0))
        else:
            # Decrease the upper bound
            self.func_values = np.minimum(self.func_values, func(self.func_values[idx] - 1.0))

# One-line description: AdaptiveBlackBoxOptimizer uses a variant of simulated annealing to refine the search space and select new points.
# Code: 