import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.search_space = np.linspace(-5.0, 5.0, self.dim)
        self.search_space = self.search_space / np.max(self.search_space)

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

        # Refine the search space based on the current function value
        if self.func_evals == 1:
            # Use a simple linear interpolation to refine the search space
            self.search_space = self.search_space / 2
            self.search_space = self.search_space / np.max(self.search_space)
        elif self.func_evals == 2:
            # Use a more sophisticated strategy based on the current function value
            self.search_space = self.search_space * np.exp(-self.func_evals / 10)
            self.search_space = self.search_space / np.max(self.search_space)
        else:
            # Use a probability-based refinement strategy
            probabilities = np.abs(self.func_values)
            cumulative_probabilities = np.cumsum(probabilities)
            idx = np.argmin(cumulative_probabilities)
            self.search_space[idx] *= 0.9
            self.search_space = self.search_space / np.max(self.search_space)

        return self.search_space

# Description: Adaptive Black Box Optimizer
# Code: 