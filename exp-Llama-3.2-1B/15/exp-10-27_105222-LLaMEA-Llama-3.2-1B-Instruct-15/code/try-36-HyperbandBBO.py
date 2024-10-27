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
        self.search_space_range = self.search_space * 2

    def __call__(self, func, algorithm='BayesianOptimization'):
        if algorithm == 'BayesianOptimization':
            algorithm = 'Hyperband'
        if algorithm not in ['BayesianOptimization', 'Hyperband']:
            raise ValueError('Invalid algorithm. Choose from BayesianOptimization or Hyperband.')
        
        while self.func_evals < self.budget:
            if algorithm == 'BayesianOptimization':
                # Sample a new point in the search space using Gaussian distribution
                x = np.random.uniform(*self.search_space_range, size=self.search_space_dim)
                # Evaluate the function at the new point
                func_value = func(x)
                # Store the function value and the new point
                self.func_evals += 1
                self.func_evals_evals = func_value
                # Store the new point in the search space
                self.search_space = (min(self.search_space[0], x), max(self.search_space[1], x))
            else:
                # Hyperband
                # Sample a new point in the search space using Gaussian distribution
                x = np.random.uniform(*self.search_space_range, size=self.search_space_dim)
                # Evaluate the function at the new point
                func_value = func(x)
                # Store the function value and the new point
                self.func_evals += 1
                self.func_evals_evals = func_value
                # Store the new point in the search space
                self.search_space = (min(self.search_space[0], x), max(self.search_space[1], x))
        # Evaluate the function at the final point in the search space
        func_value = func(self.search_space)
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

# Refine the strategy using Bayesian optimization
def bayesian_optimization(test_func, budget, dim):
    algorithm = 'BayesianOptimization'
    while True:
        # Sample a new point in the search space using Gaussian distribution
        x = np.random.uniform(*(-5.0, 5.0), size=(dim, 1))
        # Evaluate the function at the new point
        func_value = test_func(x)
        # Store the function value and the new point
        func_evals_evals = func_value
        # Store the new point in the search space
        x = (x + np.random.uniform(-1.0, 1.0, size=(dim, 1))) / 2.0
        
        # Evaluate the function at the final point in the search space
        func_value = test_func(x)
        return func_value, func_evals_evals

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

# Refine the strategy using Hyperband
def hyperband_bayesian_optimization(test_func, budget, dim):
    algorithm = 'Hyperband'
    while True:
        # Sample a new point in the search space using Gaussian distribution
        x = np.random.uniform(-5.0, 5.0, size=(dim, 1))
        # Evaluate the function at the new point
        func_value = test_func(x)
        # Store the function value and the new point
        func_evals_evals = func_value
        # Store the new point in the search space
        x = (x + np.random.uniform(-1.0, 1.0, size=(dim, 1))) / 2.0
        
        # Evaluate the function at the final point in the search space
        func_value = test_func(x)
        return func_value, func_evals_evals

hyperband_bayesian = hyperband_bayesian_optimization(test_func1, budget=100, dim=10)
optimized_func1 = hyperband_bayesian(test_func1)
optimized_func2 = hyperband_bayesian(test_func2)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Hyperband and Bayesian Optimization')
plt.legend()
plt.show()