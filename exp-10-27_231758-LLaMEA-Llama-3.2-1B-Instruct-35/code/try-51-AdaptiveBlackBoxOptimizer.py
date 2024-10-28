import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.best_func = None
        self.best_func_score = 0
        self.best_func_evals = 0

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
        if self.func_evals < self.best_func_evals:
            self.best_func_evals = self.func_evals
            self.best_func_score = self.func_values
            self.best_func = func
            # Randomly select a new function from the search space
            self.func_values = np.random.uniform(-5.0, 5.0, self.dim)

        # Evaluate the best function
        if self.func_evals >= self.best_func_evals:
            func_value = func(self.func_values)
            if np.abs(func_value - self.best_func_score) / self.best_func_score < 0.35:
                self.best_func = func
                self.best_func_score = func_value