import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.best_func = None
        self.best_score = 0.0
        self.best_idx = None
        self.min_diff = np.inf

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
        if self.func_evals == 0 and self.best_score < 0.35 * self.best_score:
            self.best_func = func
            self.best_idx = idx
            self.best_score = self.func_values[idx]
            self.min_diff = np.abs(self.func_values[idx] - self.best_score)

    def update(self, func):
        # Update the best function and its index
        if self.func_evals == 0 and self.best_score < 0.35 * self.best_score:
            self.best_func = func
            self.best_idx = np.argmin(np.abs(self.func_values))
            self.best_score = self.func_values[self.best_idx]
            self.min_diff = np.abs(self.func_values[self.best_idx] - self.best_score)

        # Refine the search space
        if self.func_evals == 0 and self.best_score < 0.35 * self.best_score:
            new_idx = np.argmin(np.abs(self.func_values))
            self.func_values[new_idx] = func(self.func_values[new_idx])
            self.func_evals -= 1
            if self.func_evals == 0:
                break

        # Refine the search space again
        if self.func_evals == 0 and self.best_score < 0.35 * self.best_score:
            new_idx = np.argmin(np.abs(self.func_values))
            self.func_values[new_idx] = func(self.func_values[new_idx])
            self.func_evals -= 1
            if self.func_evals == 0:
                break

# Description: Adaptive Black Box Optimizer
# Code: 
# ```python
def func1(x):
    return x**2

def func2(x):
    return 10 * np.sin(x)

def func3(x):
    return x**3

adaptive_optimizer = AdaptiveBlackBoxOptimizer(10, 10)
adaptive_optimizer.func1 = func1
adaptive_optimizer.func2 = func2
adaptive_optimizer.func3 = func3

adaptive_optimizer()