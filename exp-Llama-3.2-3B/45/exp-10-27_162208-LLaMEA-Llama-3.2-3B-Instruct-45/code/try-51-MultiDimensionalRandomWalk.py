import numpy as np

class MultiDimensionalRandomWalk:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.x_best = np.zeros(self.dim)
        self.f_best = float('inf')
        self.directions = [np.random.uniform(self.lower_bound, self.upper_bound, self.dim) for _ in range(10)]
        self.step_size = 1e-2

    def __call__(self, func):
        for _ in range(self.budget):
            if np.any(np.abs(self.x_best - self.lower_bound) < 1e-6) and np.any(np.abs(self.x_best - self.upper_bound) < 1e-6):
                self.x_best = self.lower_bound + np.random.uniform(0, 1, self.dim)
            else:
                for direction in self.directions:
                    x = self.x_best + direction * self.step_size
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
            # Refine strategy by changing individual lines with probability 0.45
            if np.random.rand() < 0.45:
                for i in range(self.dim):
                    if np.random.rand() < 0.5:
                        self.x_best[i] += np.random.uniform(-self.step_size, self.step_size)
                    else:
                        self.x_best[i] -= np.random.uniform(-self.step_size, self.step_size)

# Example usage
def func(x):
    return np.sum(x**2)

search = MultiDimensionalRandomWalk(budget=100, dim=10)
best_x = search(func)
print(f"Best x: {best_x}")
print(f"Best f(x): {func(best_x)}")