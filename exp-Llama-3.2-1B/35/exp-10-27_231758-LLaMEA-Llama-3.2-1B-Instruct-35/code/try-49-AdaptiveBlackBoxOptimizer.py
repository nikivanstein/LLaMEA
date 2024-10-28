import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.iterations = 0

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
            self.iterations = 0
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break
                self.iterations += 1
                if random.random() < 0.35:
                    idx = np.argmin(np.abs(self.func_values))
                    self.func_values[idx] = func(self.func_values[idx])

        # Evaluate the function with the updated values
        func(self.func_values)
        self.func_evals = 0

        # Calculate the score
        score = np.mean(np.abs(self.func_values))
        return f"AdaptiveBlackBoxOptimizer: (Score: {score:.4f}, Iterations: {self.iterations})"

# Description: Adaptive Black Box Optimizer
# Code: 