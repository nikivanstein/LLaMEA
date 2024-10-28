import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.prior = np.ones(self.dim) / self.dim

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
        self.prior = np.zeros(self.dim)
        for _ in range(100):
            idx = np.random.choice(self.dim)
            new_value = func(self.func_values[idx])
            if new_value < self.func_values[idx]:
                self.prior[idx] = 1 / (self.dim + 1)
                self.func_values[idx] = new_value
            else:
                self.prior[idx] = self.prior[idx] / (self.dim + 1)
                self.func_values[idx] = new_value

        self.func_values *= self.prior

# Description: Adaptive Black Box Optimizer
# Code: 