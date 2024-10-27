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
        self.refinement_prob = 0.45
        self.refinement_count = 0

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
                    if np.random.rand() < self.refinement_prob:
                        # Refine the direction
                        new_direction = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                        new_direction = new_direction / np.linalg.norm(new_direction)
                        new_direction = new_direction * 1e-2
                        self.directions.append(new_direction)
                        self.directions = np.array(self.directions)
                        self.directions = np.unique(self.directions, axis=0)
                        self.directions = np.sort(self.directions, axis=0)
                        self.refinement_count += 1
                        if self.refinement_count % 10 == 0:
                            self.directions = np.array([d for d in self.directions if np.any(np.abs(d) > 1e-2)])
                            self.directions = np.unique(self.directions, axis=0)
                            self.directions = np.sort(self.directions, axis=0)
                            self.refinement_count = 0
        return self.x_best, self.f_best

# Example usage
def func(x):
    return np.sum(x**2)

search = CyclicDirectionalSearch(budget=100, dim=10)
best_x, best_f = search(func)
print(f"Best x: {best_x}")
print(f"Best f(x): {best_f}")