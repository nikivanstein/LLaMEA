import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import random

class HyperbandBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim
        self.search_space_evals = 0
        self.search_space_best = None

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Sample a new point in the search space using Gaussian distribution
            x = np.random.uniform(*self.search_space, size=self.search_space_dim)
            # Evaluate the function at the new point
            func_value = func(x)
            # Store the function value and the new point
            self.func_evals += 1
            self.func_evals_evals = func_value
            # Store the new point in the search space
            self.search_space = (min(self.search_space[0], x), max(self.search_space[1], x))
            # Store the new point in the search space
            self.search_space_evals += 1
            # Store the new point in the search space
            self.search_space_best = (x, func_value)

        # Evaluate the function at the final point in the search space
        func_value = func(self.search_space_best[1])
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

# Refine the strategy using probability 0.15
def refine_strategy(individual, func, budget):
    while budget > 0:
        # Sample a new point in the search space using Gaussian distribution
        x = np.random.uniform(*individual, size=individual[1])
        # Evaluate the function at the new point
        func_value = func(x)
        # Store the function value and the new point
        budget -= 1
        individual = (x, func_value)
        # Store the new point in the search space
        if self.search_space_evals < self.budget:
            self.search_space_evals += 1
            self.search_space_best = (individual[0], func(individual[1]))
        # Store the new point in the search space
        self.search_space_evals += 1

# Run the optimization algorithm with refined strategy
refined_hyperband = HyperbandBBO(budget=100, dim=10)
optimized_func1 = refined_hyperband(test_func1)
optimized_func2 = refined_hyperband(test_func2)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Hyperband and Bayesian Optimization with Refined Strategy')
plt.legend()
plt.show()