import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.iterations = 0
        self.convergence_curve = None

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
                if random.random() < 0.35:
                    # Refine the search direction
                    self.func_values[idx] = func(self.func_values[idx] + 0.1 * (func(self.func_values[idx]) - self.func_values[idx]))
                    self.iterations += 1

        self.convergence_curve = np.abs(self.func_values - func(self.func_values))
        self.convergence_curve /= self.func_evals
        return self.convergence_curve

# Description: AdaptiveBlackBoxOptimizer: A novel metaheuristic algorithm for solving black box optimization problems.
# Code: 
# ```python
# ```python
def adaptive_black_box_optimizer(budget, dim):
    return AdaptiveBlackBoxOptimizer(budget, dim)

# Test the algorithm
func1 = lambda x: x**2
func2 = lambda x: np.sin(x)

optimizer = adaptive_black_box_optimizer(100, 10)
convergence_curve1 = optimizer(func1)
convergence_curve2 = optimizer(func2)

# Print the results
print(f"AdaptiveBlackBoxOptimizer: {convergence_curve1:.4f}")
print(f"AdaptiveBlackBoxOptimizer: {convergence_curve2:.4f}")