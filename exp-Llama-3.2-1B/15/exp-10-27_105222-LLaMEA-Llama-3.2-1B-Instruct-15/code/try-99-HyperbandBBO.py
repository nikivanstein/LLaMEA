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
        self.search_space_dim_range = (0, 10)  # Range of the search space dimension

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Sample a new point in the search space using Gaussian distribution
            x = np.random.uniform(*self.search_space_dim_range, size=self.search_space_dim)
            # Evaluate the function at the new point
            func_value = func(x)
            # Store the function value and the new point
            self.func_evals += 1
            self.func_evals_evals = func_value
            # Store the new point in the search space
            self.search_space_dim_range = (min(self.search_space_dim_range[0], x), max(self.search_space_dim_range[1], x))
        # Evaluate the function at the final point in the search space
        func_value = func(self.search_space_dim_range)
        return func_value

def bayesian_optimization(func, budget, dim, algorithm, search_space_dim_range):
    # Initialize the algorithm
    if algorithm == 'HyperbandBBO':
        return HyperbandBBO(budget, dim)
    elif algorithm == 'BayesianOptimization':
        return BayesianOptimization(func, budget, dim, search_space_dim_range)
    else:
        raise ValueError("Invalid algorithm. Supported algorithms are 'HyperbandBBO' and 'BayesianOptimization'.")

def bayesian_optimization_bbo(func, budget, dim):
    return bayesian_optimization(func, budget, dim, 'HyperbandBBO', (-5.0, 5.0))

def bayesian_optimization_hyperband(func, budget, dim):
    return bayesian_optimization(func, budget, dim, 'HyperbandBBO', (0, 10))

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

hyperband = bayesian_optimization_bbo(test_func1, 100, 10)
optimized_func1 = hyperband(test_func1)
optimized_func2 = hyperband(test_func2)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Bayesian Optimization')
plt.legend()
plt.show()

# One-line description with the main idea
# Description: Efficient Black Box Optimization using Bayesian Optimization
# Code: 