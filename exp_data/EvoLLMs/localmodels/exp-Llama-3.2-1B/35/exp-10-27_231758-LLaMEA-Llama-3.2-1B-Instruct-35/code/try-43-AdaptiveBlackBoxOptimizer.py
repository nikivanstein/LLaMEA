import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim, alpha=0.1, beta=0.5, lambda_1=0.01, lambda_2=0.01):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.alpha = alpha
        self.beta = beta
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

    def update(self, func):
        while self.func_evals > 0:
            idx = np.argmin(np.abs(self.func_values))
            self.func_values[idx] = func(self.func_values[idx])
            self.func_evals -= 1
            if self.func_evals == 0:
                break

        # Refine the strategy
        idx = np.argmin(np.abs(self.func_values))
        new_idx = idx + np.random.normal(0, 1, self.dim)
        new_func = func(self.func_values[idx])
        self.func_values[idx] = new_func
        self.func_evals = min(self.func_evals + 1, self.budget)
        self.update(func)

    def run(self, func, max_iter=100, tol=1e-6):
        for _ in range(max_iter):
            self.update(func)
            if np.linalg.norm(self.func_values - func(self.func_values)) < tol:
                break
        return self.func_values

# Description: Adaptive Black Box Optimizer
# Code: 
# ```python
# AdaptiveBlackBoxOptimizer: An adaptive black box optimizer that refines its strategy based on the individual lines of the selected solution.
# ```python