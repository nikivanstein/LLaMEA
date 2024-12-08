import numpy as np
import random
from scipy.optimize import differential_evolution

class DABU:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

    def differential_evolution(self, func, bounds):
        res = differential_evolution(func, bounds)
        return res.fun, res.x

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

dabu = DABU(1000, 2)  # 1000 function evaluations, 2 dimensions
res, x = dabu.differential_evolution(test_function, [(-5.0, 5.0), (-5.0, 5.0)])
print(f"Optimal solution: {x}")
print(f"Optimal function value: {res}")

# Refine the strategy by changing the number of iterations and the step size
dabu = DABU(1000, 2)
dabu.search_space = np.linspace(-3.0, 3.0, 3)
res, x = dabu.differential_evolution(test_function, [(-3.0, 3.0), (-3.0, 3.0)])
print(f"Optimal solution: {x}")
print(f"Optimal function value: {res}")