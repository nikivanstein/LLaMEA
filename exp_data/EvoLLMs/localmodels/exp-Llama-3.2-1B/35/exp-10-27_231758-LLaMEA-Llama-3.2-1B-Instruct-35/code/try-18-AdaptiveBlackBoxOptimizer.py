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

        # Refine the strategy using probability 0.35
        if random.random() < 0.35:
            # Increase the number of evaluations
            self.func_evals *= 2
            # Increase the search space
            self.dim *= 2
        else:
            # Decrease the number of evaluations
            self.func_evals //= 2
            # Decrease the search space
            self.dim //= 2

# One-line description: AdaptiveBlackBoxOptimizer: A novel metaheuristic algorithm for solving black box optimization problems by dynamically adjusting the search space and evaluation rate.
# Code: 