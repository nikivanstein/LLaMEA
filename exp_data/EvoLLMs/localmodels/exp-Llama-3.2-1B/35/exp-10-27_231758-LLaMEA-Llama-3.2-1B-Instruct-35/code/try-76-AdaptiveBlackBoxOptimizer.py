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

        # Refine the strategy using adaptive sampling and evolutionary strategies
        self.func_values = differential_evolution(lambda x: -x, [self.func_values], x0=np.array([self.func_values]))
        return self

    def optimize_function(self, func, max_iter=1000):
        return func(self.func_values)

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

optimizer = AdaptiveBlackBoxOptimizer(budget=1000, dim=2)
print(optimizer.optimize_function(func))