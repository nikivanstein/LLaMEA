import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class HyperbandBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim

    def __call__(self, func, algorithm="bbo"):
        if algorithm == "bbo":
            return self.budget(self.func_evals, self.search_space, func)
        elif algorithm == "hyperband":
            return self.budget(self.func_evals, self.search_space, func, self.dim)
        else:
            raise ValueError("Invalid algorithm. Please choose from 'bbo' or 'hyperband'.")

    def budget(self, func_evals, search_space, func, dim):
        # Sample a new point in the search space using Gaussian distribution
        x = np.random.uniform(*search_space, size=dim)
        # Evaluate the function at the new point
        func_value = func(x)
        # Store the function value and the new point
        self.func_evals += 1
        self.func_evals_evals = func_value
        # Store the new point in the search space
        self.search_space = (min(self.search_space[0], x), max(self.search_space[1], x))
        return func_value

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

hyperband = HyperbandBBO(budget=100, dim=10)
optimized_func1 = hyperband(test_func1)
optimized_func2 = hyperband(test_func2)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Hyperband and Bayesian Optimization')
plt.legend()
plt.show()