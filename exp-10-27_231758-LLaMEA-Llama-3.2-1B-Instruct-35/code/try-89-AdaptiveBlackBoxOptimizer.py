import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

    def __call__(self, func):
        if self.func_values is None:
            # Use a simple adaptive strategy to avoid exploring the entire search space
            # at once. This strategy will be refined later.
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                # Use the Kuhn-Tucker conditions to refine the search space
                idx = np.argmin(np.abs(self.func_values))
                # Update the value of the function at the current point
                self.func_values[idx] = func(self.func_values[idx])
                # Update the evaluation count
                self.func_evals -= 1
                # If all evaluations have been completed, break the loop
                if self.func_evals == 0:
                    break

    def optimize(self, func, max_iter=1000, tol=1e-6):
        # Use the Kuhn-Tucker conditions to refine the search space
        for _ in range(max_iter):
            # Update the function value at each dimension
            for dim in range(self.dim):
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[dim] = func(self.func_values[dim])
            # Update the evaluation count
            self.func_evals = 0
            # Check if all evaluations have been completed
            if self.func_evals == 0:
                break

# Description: AdaptiveBlackBoxOptimizer: A novel metaheuristic algorithm for solving black box optimization problems.
# Code: 