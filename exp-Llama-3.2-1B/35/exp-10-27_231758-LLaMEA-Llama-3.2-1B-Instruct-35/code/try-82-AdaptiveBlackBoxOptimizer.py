import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.learning_rate = 0.1
        self.adaptive_sampling = True
        self.adaptive_learning_rate = 0.01
        self.max_iter = 1000

    def __call__(self, func, num_samples):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            if self.adaptive_sampling:
                while self.func_evals > 0:
                    idx = np.argmin(np.abs(self.func_values))
                    self.func_values[idx] = func(self.func_values[idx])
                    self.func_evals -= 1
                    if self.func_evals == 0:
                        break
            else:
                idx = np.random.randint(0, self.dim)
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

        if self.adaptive_sampling and self.func_evals > 0:
            self.learning_rate = np.min([0.1, self.learning_rate * 0.9, 0.01])
        else:
            self.learning_rate = 0.1

        return self.func_values

# One-line description with main idea
# Adaptive Black Box Optimizer with Adaptive Sampling and Adaptive Learning Rate
# The algorithm uses adaptive sampling and adaptive learning rate to refine its strategy in the optimization process