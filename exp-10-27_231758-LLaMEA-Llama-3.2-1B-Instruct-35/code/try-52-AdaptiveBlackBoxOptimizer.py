import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.best_func = None
        self.best_score = float('-inf')
        self.best_idx = None

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

        # Refine the strategy
        if self.func_evals == 1 and random.random() < 0.35:
            self.func_evals += 1
            while self.func_evals > 1:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 1:
                    break

        # Update the best function
        if self.func_evals > 0:
            self.best_func = func
            self.best_score = np.max(np.abs(self.func_values))
            self.best_idx = np.argmin(np.abs(self.func_values))

        return self.best_func

# Description: AdaptiveBlackBoxOptimizer
# Code: 