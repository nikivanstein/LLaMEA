import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.alpha = 0.5  # probability of refining the individual lines
        self.beta = 0.5  # probability of not refining the individual lines

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                if np.random.rand() < self.alpha:
                    # Refine the individual line
                    self.func_values[idx] = func(self.func_values[idx])
                else:
                    # Do not refine the individual line
                    break
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

# Description: AdaptiveBlackBoxOptimizer: An adaptive black box optimization algorithm that refines individual lines with a probability of 0.5.
# Code: 
# ```python
# ```python
# # Initialize the AdaptiveBlackBoxOptimizer with a budget and dimension
optimizer = AdaptiveBlackBoxOptimizer(1000, 10)

# Define a function to be optimized
def func(x):
    return np.sin(x)

# Optimize the function using the AdaptiveBlackBoxOptimizer
optimizer(func)