import numpy as np
from scipy.optimize import differential_evolution

class NovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.search_space_min = -5.0
        self.search_space_max = 5.0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

    def optimize(self, func):
        # Refine the search space using probability 0.3
        search_space = self.search_space.copy()
        if np.random.rand() < 0.3:
            # Increase the upper bound by 10%
            self.search_space_max *= 1.1
        else:
            # Decrease the upper bound by 10%
            self.search_space_max /= 1.1
        if np.random.rand() < 0.3:
            # Increase the lower bound by 10%
            self.search_space_min *= 1.1
        else:
            # Decrease the lower bound by 10%
            self.search_space_min /= 1.1
        return self.optimize_func(func, search_space)

    def optimize_func(self, func, search_space):
        # Perform differential evolution to optimize the function
        res = differential_evolution(func, search_space)
        return res.fun

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

novel_metaheuristic = NovelMetaheuristic(1000, 2)  # 1000 function evaluations, 2 dimensions
print(novel_metaheuristic(test_function))  # prints a random value between -10 and 10