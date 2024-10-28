import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveDDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.local_search = False

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

    def adaptive_local_search(self, func):
        while self.func_evaluations < self.budget:
            x, _ = differential_evolution(func, self.search_space)
            if np.abs(x[0]**2 + x[1]**2) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
            if self.local_search:
                self.search_space = np.linspace(-5.0, 5.0, self.dim)
            else:
                self.search_space = np.linspace(-5.0, 5.0, self.dim) * 2
            self.local_search = not self.local_search