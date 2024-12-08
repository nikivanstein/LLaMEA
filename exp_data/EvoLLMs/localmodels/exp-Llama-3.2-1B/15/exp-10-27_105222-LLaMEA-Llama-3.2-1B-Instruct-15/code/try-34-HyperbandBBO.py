import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class HyperbandBBO:
    def __init__(self, budget, dim, algorithm='Bayesian'):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim
        self.algorithm = algorithm
        if algorithm == 'Bayesian':
            self.func_evals = 0
            self.search_space = (-5.0, 5.0)
            self.search_space_dim = self.dim
        elif algorithm == 'Hyperband':
            self.func_evals = 0
            self.search_space = (-5.0, 5.0)
            self.search_space_dim = self.dim
        else:
            raise ValueError("Invalid algorithm. Choose from 'Bayesian' or 'Hyperband'.")

    def __call__(self, func):
        if self.algorithm == 'Bayesian':
            # Bayesian Optimization
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
        elif self.algorithm == 'Hyperband':
            # Hyperband Optimization
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
        # Evaluate the function at the final point in the search space
        func_value = func(self.search_space)
        return func_value

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

hyperband = HyperbandBBO(budget=100, dim=10, algorithm='Bayesian')
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

# Refine the strategy
def refine_strategy(individual, func, budget):
    if func(individual) > func(refine_strategy(individual, func, budget - 1)):
        return refine_strategy(individual, func, budget - 1)
    else:
        return individual

# Apply the refined strategy
optimized_func1 = refine_strategy(optimized_func1, test_func1, 50)
optimized_func2 = refine_strategy(optimized_func2, test_func2, 50)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Refined Test Function 1', 'Refined Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Hyperband and Bayesian Optimization')
plt.legend()
plt.show()