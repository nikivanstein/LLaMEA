import numpy as np
from scipy.optimize import differential_evolution, shgo

class LeverageExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.x_best = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
        self.f_best = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            # Explore
            x_explore = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
            f_explore = func(x_explore)

            # Adaptive exploration strategy
            if np.random.rand() < 0.5:
                x_explore = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)

            # Local search using SHGO for better exploration
            x_dev = shgo(func, bounds=[(self.lower_bound, self.upper_bound) for _ in range(self.dim)])
            f_dev = func(x_dev.x)

            # Leverage the best exploration point
            if f_explore < f_dev:
                self.x_best = x_explore
                self.f_best = f_explore

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

le = LeverageExploration(budget=100, dim=2)
le(func)
print("Best point:", le.x_best)
print("Best function value:", le.f_best)