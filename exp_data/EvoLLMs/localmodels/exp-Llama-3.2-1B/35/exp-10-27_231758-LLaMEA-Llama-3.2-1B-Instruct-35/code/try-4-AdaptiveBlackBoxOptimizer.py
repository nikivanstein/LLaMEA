import numpy as np
from scipy.optimize import differential_evolution
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

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

        # Refine the strategy using adaptive sampling
        if random.random() < 0.35:
            self.func_values = np.random.uniform(-5.0, 5.0, self.dim)
            self.func_evals = 1
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

        # Use evolutionary strategies to further optimize the function values
        bounds = [(-5.0, 5.0)] * self.dim
        res = differential_evolution(lambda x: -x, bounds, args=(func, self.func_values))
        self.func_values = res.x

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

optimizer = AdaptiveBlackBoxOptimizer(10, 2)
optimizer(func)