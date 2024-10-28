import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.explore_rate = 0.1
        self.exploration_threshold = 0.05
        self.convergence_threshold = 0.01

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

                if random.random() < self.explore_rate:
                    # Randomly select a new function value
                    idx = np.random.randint(0, self.dim)
                    func(self.func_values[idx])
                else:
                    # Use the current function value with a fixed probability
                    func(self.func_values[idx])

                if self.func_evals > self.convergence_threshold:
                    break

                if np.abs(self.func_values[idx] - self.func_values[idx - 1]) < self.convergence_threshold:
                    # If the function value converges, reduce the exploration rate
                    self.explore_rate *= self.explore_rate
                    if self.explore_rate < self.exploration_threshold:
                        self.explore_rate = self.exploration_threshold