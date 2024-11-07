import numpy as np
import random

class LeverageEntropy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.f_best = None
        self.x_best = None
        self.f_best_val = float('inf')
        self.entropy = 0.0

    def _generate_random_point(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

    def _calculate_entropy(self, x):
        entropy = 0.0
        for i in range(self.dim):
            if x[i]!= self.lower_bound and x[i]!= self.upper_bound:
                entropy += 1 / np.log(2 * np.pi * np.sqrt(1 + (x[i] - self.lower_bound) ** 2))
        return entropy

    def __call__(self, func):
        if self.f_best is None or self.f_best_val > func(self.x_best):
            self.f_best = func(self.x_best)
            self.x_best = self.x_best
            self.f_best_val = self.f_best

        for _ in range(self.budget - 1):
            x = self._generate_random_point()
            f = func(x)
            entropy = self._calculate_entropy(x)

            if f < self.f_best:
                self.f_best = f
                self.x_best = x

            if self.f_best_val - f < 1e-3:
                entropy -= entropy / 2

            self.entropy += entropy

        self.entropy = max(0.0, self.entropy - 0.1)

        if self.f_best_val > func(self.x_best):
            self.f_best = func(self.x_best)
            self.x_best = self.x_best

        return self.f_best

# Example usage
def func(x):
    return np.sum(x ** 2)

budget = 100
dim = 10
leverage_entropy = LeverageEntropy(budget, dim)
for _ in range(100):
    func(leverage_entropy())