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

    def __call__(self, func, algorithm, dim):
        # Evaluate the function using the provided algorithm
        func_value = algorithm(func, self.search_space_dim)
        # Store the function value and the new point
        self.func_evals += 1
        self.func_evals_evals = func_value
        # Store the new point in the search space
        self.search_space = (min(self.search_space[0], func_value), max(self.search_space[1], func_value))
        # Evaluate the function at the final point in the search space
        func_value = func(self.search_space)
        return func_value

# Hyperband Algorithm
class Hyperband:
    def __init__(self, budget, dim, algorithm, dim_range):
        self.budget = budget
        self.dim = dim
        self.dim_range = dim_range
        self.func_evals = 0
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim
        self.algorithm = algorithm

    def __call__(self, func, algorithm, dim):
        # Generate a new point in the search space using Gaussian distribution
        x = np.random.uniform(*self.search_space, size=self.search_space_dim)
        # Evaluate the function at the new point using the provided algorithm
        func_value = self.algorithm(func, x)
        # Store the function value and the new point
        self.func_evals += 1
        self.func_evals_evals = func_value
        # Store the new point in the search space
        self.search_space = (min(self.search_space[0], func_value), max(self.search_space[1], func_value))
        # Evaluate the function at the final point in the search space
        func_value = func(self.search_space)
        return func_value

# Bayesian Optimization Algorithm
class BayesianOptimization:
    def __init__(self, budget, dim, algorithm, dim_range):
        self.budget = budget
        self.dim = dim
        self.dim_range = dim_range
        self.func_evals = 0
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim
        self.algorithm = algorithm

    def __call__(self, func, algorithm, dim):
        # Generate a new point in the search space using Gaussian distribution
        x = np.random.uniform(*self.search_space, size=self.search_space_dim)
        # Evaluate the function at the new point using the provided algorithm
        func_value = self.algorithm(func, x)
        # Store the function value and the new point
        self.func_evals += 1
        self.func_evals_evals = func_value
        # Store the new point in the search space
        self.search_space = (min(self.search_space[0], func_value), max(self.search_space[1], func_value))
        # Evaluate the function at the final point in the search space
        func_value = func(self.search_space)
        return func_value

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

hyperband = Hyperband(budget=100, dim=10, algorithm=HyperbandBBO, dim_range=(1, 10))
optimized_func1 = hyperband(test_func1, Hyperband, dim=10)
optimized_func2 = hyperband(test_func2, BayesianOptimization, dim=10)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Hyperband and Bayesian Optimization')
plt.legend()
plt.show()

# Refine the strategy using probability 0.15
def refine_strategy(func, algorithm, dim, budget, dim_range):
    # Evaluate the function using the provided algorithm
    func_value = algorithm(func, dim)
    # Store the function value and the new point
    self.func_evals += 1
    self.func_evals_evals = func_value
    # Store the new point in the search space
    self.search_space = (min(self.search_space[0], func_value), max(self.search_space[1], func_value))
    # Evaluate the function at the final point in the search space
    func_value = func(self.search_space)
    # Refine the strategy using probability 0.15
    self.func_evals_evals *= 0.85
    self.func_evals_evals = func_value
    return func_value

refined_func1 = refine_strategy(test_func1, HyperbandBBO, dim=10, budget=100, dim_range=(1, 10))
refined_func2 = refine_strategy(test_func2, BayesianOptimization, dim=10, budget=100, dim_range=(1, 10))

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([refined_func1, refined_func2], label=['Refined Test Function 1', 'Refined Test Function 2'])
plt.xlabel('Refined Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Hyperband and Bayesian Optimization')
plt.legend()
plt.show()