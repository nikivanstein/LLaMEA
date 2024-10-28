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

    def __str__(self):
        return f"AdaptiveBlackBoxOptimizer: Adaptive Black Box Optimization with adaptive budget and dimensionality"

    def adaptive_budget(self, func_evals):
        # Refine the adaptive budget based on the number of evaluations
        if func_evals < 100:
            return 10
        elif func_evals < 500:
            return 50
        else:
            return 100

    def adaptive_dimensionality(self, func_dim):
        # Refine the adaptive dimensionality based on the number of evaluations
        if func_dim < 10:
            return 2
        elif func_dim < 50:
            return 5
        else:
            return 10

    def __str__(self):
        return f"AdaptiveBlackBoxOptimizer: Adaptive Black Box Optimization with adaptive budget ({self.adaptive_budget(self.func_evals)}) and dimensionality ({self.adaptive_dimensionality(self.dim)})"

# Description: Adaptive Black Box Optimization with adaptive budget and dimensionality refinement
# Code: 
# ```python
def func(x):
    return x**2 + 0.5 * random.random() * (x**4 - 2 * x**2 + 1)

def adaptive_black_box_optimizer(budget, dim):
    optimizer = AdaptiveBlackBoxOptimizer(budget, dim)
    optimizer.func_values = np.zeros(dim)
    for _ in range(budget):
        func_value = func(optimizer.func_values)
        optimizer.func_values = func_value
    return optimizer

# Evaluate the optimizer
optimizer = adaptive_black_box_optimizer(1000, 10)
optimizer.func_values = np.random.rand(10)
print(optimizer)

# Refine the optimizer
budget = optimizer.adaptive_budget(1000)
dim = optimizer.adaptive_dimensionality(10)
optimizer = adaptive_black_box_optimizer(budget, dim)
print(optimizer)