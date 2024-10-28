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

        # Refine the strategy
        if self.func_evals < self.budget // 2:
            # Exploration-exploitation trade-off
            epsilon = random.uniform(0, 1)
            if epsilon < 0.5:
                # Exploration: increase the number of evaluations
                self.func_evals *= 2
            else:
                # Exploitation: decrease the number of evaluations
                self.func_evals //= 2

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.optimizer = AdaptiveBlackBoxOptimizer(budget, dim)

    def __call__(self, func):
        self.optimizer(func)

    def score(self):
        return self.optimizer.func_values.mean()

# Description: Adaptive Black Box Optimizer with adaptive search strategy
# Code: 