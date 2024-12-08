import numpy as np
from scipy.optimize import differential_evolution

class DynamicAdaptiveBoundedOptimization:
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

    def adaptive_strategy(self, func):
        # Refine the search space based on the function value
        if np.abs(func(self.search_space)) < 1e-6:
            # If the function value is close to zero, expand the search space
            self.search_space = np.linspace(-5.0, 5.0, self.dim * 2)
        else:
            # If the function value is not close to zero, contract the search space
            self.search_space = np.linspace(-5.0, 5.0, self.dim)

    def bounded_search(self, func, initial_point, bounds):
        # Perform a bounded search using differential evolution
        result = differential_evolution(func, bounds, x0=initial_point)
        return result

    def run(self, func, initial_point, bounds):
        # Run the optimization algorithm
        self.adaptive_strategy(func)
        bounded_search_result = self.bounded_search(func, initial_point, bounds)
        # Refine the search space based on the bounded search result
        if bounded_search_result.x[0] < 0:
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        else:
            self.search_space = np.linspace(-5.0, 5.0, self.dim * 2)
        return bounded_search_result

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

dabu = DynamicAdaptiveBoundedOptimization(1000, 2)  # 1000 function evaluations, 2 dimensions
print(dabu(test_function))  # prints a random value between -10 and 10