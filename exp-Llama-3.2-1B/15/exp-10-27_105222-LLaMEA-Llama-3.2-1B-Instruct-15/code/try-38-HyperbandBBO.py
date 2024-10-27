import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class HyperbandBBO:
    def __init__(self, budget, dim, algorithm, strategy):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim
        self.algorithm = algorithm
        self.strategy = strategy
        self.func_evals = 0
        self.func_evals_evals = 0

    def __call__(self, func, initial_individual=None):
        if initial_individual is None:
            # Sample a new point in the search space using Gaussian distribution
            x = np.random.uniform(*self.search_space, size=self.search_space_dim)
        else:
            # Use the previous best individual
            x = initial_individual
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

hyperband = HyperbandBBO(budget=100, dim=10, algorithm="Hyperband", strategy="BayesianOptimization")
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
def refine_strategy(individual, func, budget, dim, algorithm, strategy):
    if strategy == "BayesianOptimization":
        # Bayesian Optimization
        new_individual = individual
        while True:
            # Evaluate the function at the new individual
            func_value = func(new_individual)
            # Store the function value and the new individual
            new_individual = individual
            self.func_evals_evals = func_value
            # Store the new individual in the search space
            self.search_space = (min(self.search_space[0], new_individual), max(self.search_space[1], new_individual))
            # Check if the budget is exceeded
            if self.func_evals_evals >= budget:
                break
    elif strategy == "Hyperband":
        # Hyperband
        while True:
            # Sample a new point in the search space using Gaussian distribution
            new_individual = np.random.uniform(*self.search_space, size=self.search_space_dim)
            # Evaluate the function at the new individual
            func_value = func(new_individual)
            # Store the function value and the new individual
            new_individual = new_individual
            self.func_evals_evals = func_value
            # Store the new individual in the search space
            self.search_space = (min(self.search_space[0], new_individual), max(self.search_space[1], new_individual))
            # Check if the budget is exceeded
            if self.func_evals_evals >= budget:
                break
    else:
        raise ValueError("Invalid strategy")

# Refine the strategy
refine_strategy(optimized_func1, test_func1, 100, 10, "Hyperband", "BayesianOptimization")

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Hyperband and Bayesian Optimization')
plt.legend()
plt.show()