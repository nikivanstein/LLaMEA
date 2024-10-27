import numpy as np
import random

class ProbabilisticDirectionalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.x_best = np.zeros(self.dim)
        self.f_best = float('inf')
        self.directions = [np.random.uniform(self.lower_bound, self.upper_bound, self.dim) for _ in range(10)]
        self.probability = 0.45

    def __call__(self, func):
        for _ in range(self.budget):
            if np.any(np.abs(self.x_best - self.lower_bound) < 1e-6) and np.any(np.abs(self.x_best - self.upper_bound) < 1e-6):
                self.x_best = self.lower_bound + np.random.uniform(0, 1, self.dim)
            else:
                for direction in self.directions:
                    x = self.x_best + direction * 1e-2
                    f = func(x)
                    if f < self.f_best:
                        self.x_best = x
                        self.f_best = f
            if np.random.rand() < self.probability:
                # Refine strategy by changing individual lines
                for _ in range(int(self.budget * self.probability)):
                    # Change x_best by adding a random direction
                    x = self.x_best + np.random.uniform(-1, 1, self.dim)
                    f = func(x)
                    if f < self.f_best:
                        self.x_best = x
                        self.f_best = f
                    # Change direction by adding a new random direction
                    direction = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                    f = func(self.x_best + direction * 1e-2)
                    if f < self.f_best:
                        self.x_best = self.x_best + direction * 1e-2
                        self.f_best = f
            self.directions = [direction for direction in self.directions if np.any(np.abs(direction) > 1e-2)]
            self.directions.append(np.random.uniform(self.lower_bound, self.upper_bound, self.dim))
            self.directions = np.array(self.directions)
            self.directions = np.unique(self.directions, axis=0)
            self.directions = np.sort(self.directions, axis=0)

# Example usage
def func(x):
    return np.sum(x**2)

search = ProbabilisticDirectionalSearch(budget=100, dim=10)
best_x = search(func)
print(f"Best x: {best_x}")
print(f"Best f(x): {func(best_x)}")