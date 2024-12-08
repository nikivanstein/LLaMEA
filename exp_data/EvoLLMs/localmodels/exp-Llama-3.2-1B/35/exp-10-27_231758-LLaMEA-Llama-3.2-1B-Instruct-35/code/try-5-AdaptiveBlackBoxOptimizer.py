import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.bounds = None
        self.bounds_strategy = None
        self.bounds_strategy_params = None
        self.score = 0.0

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
        self.refine_strategy()

    def refine_strategy(self):
        # 0.35: Use a simple strategy (e.g., 10% increase/decrease in the lower bound)
        if self.bounds_strategy =='simple':
            self.bounds_strategy_params = random.uniform(0.1, 0.3)
            self.bounds = self.bounds + self.bounds_strategy_params
        # 0.35: Use a more sophisticated strategy (e.g., use the current value of the function)
        elif self.bounds_strategy == 'value':
            self.bounds = self.bounds + self.func_values
        # 0.35: Use a random strategy (e.g., use a random direction)
        elif self.bounds_strategy == 'random':
            self.bounds = self.bounds + np.random.uniform(-self.bounds[0], self.bounds[1], self.dim)

    def __str__(self):
        return f"AdaptiveBlackBoxOptimizer(budget={self.budget}, dim={self.dim})"

# Description: AdaptiveBlackBoxOptimizer
# Code: 
# ```python
# AdaptiveBlackBoxOptimizer(budget=100, dim=10)