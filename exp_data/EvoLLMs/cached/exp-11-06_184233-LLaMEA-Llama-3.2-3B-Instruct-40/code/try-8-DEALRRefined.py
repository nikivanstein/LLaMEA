import numpy as np
import random
from scipy.optimize import differential_evolution

class DEALRRefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=(budget, dim))
        self.f = np.zeros(budget)
        self.pop = np.zeros((budget, dim))
        self.LR = 0.5 + 0.1 * np.random.uniform(-0.1, 0.1, size=1)
        self.GR = 0.5 + 0.1 * np.random.uniform(-0.1, 0.1, size=1)

    def __call__(self, func):
        for i in range(self.budget):
            if i > 0:
                self.pop[i] = self.f[i-1] + np.random.uniform(-1, 1, size=self.dim)
            f_i = func(self.pop[i])
            if f_i < self.f[i]:
                self.x[i] = self.pop[i]
                self.f[i] = f_i
        return self.x[np.argmin(self.f)], np.min(self.f)

    def optimize(self, func):
        bounds = [(-5.0, 5.0)] * self.dim
        result = differential_evolution(func, bounds, x0=self.x, args=(), maxiter=self.budget, tol=1e-6)
        return result.x[0], result.fun

# Example usage:
def func(x):
    return np.sum(x**2)

dealr_refined = DEALRRefined(budget=100, dim=10)
x, f = dealr_refined.optimize(func)
print(f'Optimal solution: x = {x}, f = {f}')

# Test on BBOB test suite
from blackboxoptimalsuite import BBOB

bbo = BBOB(24, 1)
for i in range(24):
    func = bbo.get_function(i)
    x, f = dealr_refined.optimize(func)
    print(f'Optimal solution for function {i}: x = {x}, f = {f}')