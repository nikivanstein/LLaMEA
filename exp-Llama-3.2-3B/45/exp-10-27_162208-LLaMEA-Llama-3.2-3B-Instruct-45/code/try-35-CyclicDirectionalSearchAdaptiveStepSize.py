import numpy as np
import random

class CyclicDirectionalSearchAdaptiveStepSize:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.x_best = np.zeros(self.dim)
        self.f_best = float('inf')
        self.directions = [np.random.uniform(self.lower_bound, self.upper_bound, self.dim) for _ in range(10)]
        self.step_sizes = [1e-2]
        self.adaptation_rate = 0.45

    def __call__(self, func):
        for _ in range(self.budget):
            if np.any(np.abs(self.x_best - self.lower_bound) < 1e-6) and np.any(np.abs(self.x_best - self.upper_bound) < 1e-6):
                self.x_best = self.lower_bound + np.random.uniform(0, 1, self.dim)
            else:
                for direction in self.directions:
                    x = self.x_best + direction * self.step_sizes[0]
                    f = func(x)
                    if f < self.f_best:
                        self.x_best = x
                        self.f_best = f
                if self.f_best < func(self.x_best):
                    self.x_best = self.x_best
                    self.f_best = self.f_best
                self.directions = [direction for direction in self.directions if np.any(np.abs(direction) > 1e-2)]
                self.directions.append(np.random.uniform(self.lower_bound, self.upper_bound, self.dim))
                self.directions = np.array(self.directions)
                self.directions = np.unique(self.directions, axis=0)
                self.directions = np.sort(self.directions, axis=0)
                if len(self.directions) > 10:
                    self.directions = self.directions[:10]
            if self.f_best < func(self.x_best):
                self.x_best = self.x_best
                self.f_best = self.f_best
            if random.random() < self.adaptation_rate:
                self.step_sizes.append(self.step_sizes[-1] * 1.1)
                self.step_sizes = np.array(self.step_sizes)
                self.step_sizes = np.unique(self.step_sizes, axis=0)
                self.step_sizes = np.sort(self.step_sizes, axis=0)
            else:
                self.step_sizes = [size * 0.9 for size in self.step_sizes]

# Example usage
def func(x):
    return np.sum(x**2)

search = CyclicDirectionalSearchAdaptiveStepSize(budget=100, dim=10)
best_x = search(func)
print(f"Best x: {best_x}")
print(f"Best f(x): {func(best_x)}")