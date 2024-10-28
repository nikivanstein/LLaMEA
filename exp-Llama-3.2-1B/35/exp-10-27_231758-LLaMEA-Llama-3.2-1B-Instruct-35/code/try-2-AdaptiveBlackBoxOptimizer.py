import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.best_func = None
        self.best_score = 0.0
        self.best_idx = None

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
        # Update the best function and its score
        self.best_func = func
        self.best_score = np.mean(self.func_values)
        if self.best_score > self.best_score_old:
            self.best_idx = np.argmin(np.abs(self.func_values))
            self.best_func = func
            self.best_score_old = self.best_score

        # Change the individual lines of the selected solution to refine its strategy
        # 1. Explore the search space
        if np.random.rand() < 0.2:
            self.func_values = np.random.uniform(-5.0, 5.0, self.dim)
        # 2. Conserve the best function
        if np.random.rand() < 0.3:
            self.func_values[self.best_idx] = self.best_func(self.func_values[self.best_idx])

# One-line description with the main idea:
# AdaptiveBlackBoxOptimizer: A novel metaheuristic algorithm that refines its strategy using probability 0.35 to improve its performance in solving black box optimization problems.

# BBOB test suite of 24 noiseless functions
# The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions.
# The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.

# Example usage:
# Create an instance of AdaptiveBlackBoxOptimizer with a budget of 100 and a dimension of 5
optimizer = AdaptiveBlackBoxOptimizer(100, 5)

# Optimize a black box function using the optimizer
# func = lambda x: x**2
# optimizer(func)

# Print the updated best function and its score
print(optimizer.func_values)
print(optimizer.best_score)