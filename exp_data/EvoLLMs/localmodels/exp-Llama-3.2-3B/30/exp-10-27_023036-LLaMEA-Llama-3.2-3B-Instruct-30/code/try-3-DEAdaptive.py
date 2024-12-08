import numpy as np
from scipy.optimize import differential_evolution

class DEAdaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x0 = np.random.uniform(-5.0, 5.0, size=(budget, dim))
        self.f_best = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            res = differential_evolution(func, [(-5.0, 5.0)] * self.dim)
            self.x0 = np.vstack((self.x0, res.x))
            self.f_best = min(self.f_best, res.fun)
            if np.random.rand() < 0.3:
                idx = np.random.randint(0, self.budget)
                self.x0[idx] = res.x
                self.f_best = min(self.f_best, res.fun)
        return self.x0, self.f_best

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
solver = DEAdaptive(budget, dim)
x_opt, f_opt = solver(func)
print(f"Optimal solution: x = {x_opt}, f(x) = {f_opt}")