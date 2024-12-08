import numpy as np
from scipy.optimize import minimize

class ACR:
    def __init__(self, budget, dim, initial_strategy, refinement_factor):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.initial_strategy = initial_strategy
        self.refinement_factor = refinement_factor

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

    def optimize(self, func):
        # Initialize strategy with the initial strategy
        strategy = self.initial_strategy
        # Refine strategy based on the number of function evaluations
        for _ in range(self.budget - 1):
            # Evaluate the function at the current strategy
            func_value = func(strategy)
            # Refine the strategy based on the convergence rate
            if np.abs(func_value) < 1e-6:
                break
            # Update the strategy based on the refinement factor
            strategy = strategy + self.refinement_factor * np.random.uniform(-1, 1, self.dim)
        return strategy

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

def optimize_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

dabu = ACR(1000, 2, test_function, 0.5)  # 1000 function evaluations, 2 dimensions
print(dabu(test_function))  # prints a random value between -10 and 10

# Update the solution
dabu = ACR(1000, 2, dabu.optimize(test_function), 0.5)  # 1000 function evaluations, 2 dimensions
print(dabu(test_function))  # prints a refined value between -10 and 10