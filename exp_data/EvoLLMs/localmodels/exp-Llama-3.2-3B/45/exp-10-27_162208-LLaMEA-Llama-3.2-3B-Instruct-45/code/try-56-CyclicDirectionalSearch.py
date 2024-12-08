import numpy as np
import random

class CyclicDirectionalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.x_best = np.zeros(self.dim)
        self.f_best = float('inf')
        self.directions = [np.random.uniform(self.lower_bound, self.upper_bound, self.dim) for _ in range(10)]
        self.refine_prob = 0.45

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
                if random.random() < self.refine_prob:
                    # Refine the current solution
                    for _ in range(int(self.budget * self.refine_prob)):
                        x = self.x_best + np.random.uniform(-1e-2, 1e-2, self.dim)
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

# Example usage
def func(x):
    return np.sum(x**2)

search = CyclicDirectionalSearch(budget=100, dim=10)
best_x = search(func)
print(f"Best x: {best_x}")
print(f"Best f(x): {func(best_x)}")