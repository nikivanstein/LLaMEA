import numpy as np
from scipy.optimize import differential_evolution

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

        # Refine the strategy by adjusting the number of evaluations and the search space
        if self.func_evals > self.budget // 5:
            self.func_evals = self.budget // 5
            self.func_values = np.linspace(-5.0, 5.0, self.dim)

        # Use differential evolution to find the optimal solution
        res = differential_evolution(lambda x: -x, [(self.func_values - 0.5), (self.func_values + 0.5)], x0=self.func_values)
        self.func_values = res.x

    def update(self, func):
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

        # Refine the strategy by adjusting the number of evaluations and the search space
        if self.func_evals > self.budget // 5:
            self.func_evals = self.budget // 5
            self.func_values = np.linspace(-5.0, 5.0, self.dim)

# Description: Adaptive Black Box Optimization using Differential Evolution
# Code: 