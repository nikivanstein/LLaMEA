import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class HyperbandBBO:
    def __init__(self, budget, dim, alpha, beta):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.func_evals = 0
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim
        self.search_space_bounds = self.search_space

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
            self.search_space = (min(self.search_space_bounds[0], x), max(self.search_space_bounds[1], x))
        # Evaluate the function at the final point in the search space
        func_value = func(self.search_space)
        return func_value

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

hyperband = HyperbandBBO(budget=100, dim=10, alpha=0.15, beta=0.15)
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

# Refine the strategy using the current population of algorithms
def refine_strategy(algorithm):
    # Select the algorithm with the highest score
    best_algorithm = max(algorithm, key=lambda x: x.score)
    # Update the best algorithm's strategy
    best_algorithm.search_space_bounds = (best_algorithm.search_space_bounds[0] * 1.1, best_algorithm.search_space_bounds[1] * 0.9)
    return best_algorithm

# Add the refined strategy to the current population
refined_hyperband = refine_strategy([hyperband])
optimized_func1 = refined_hyperband(test_func1)
optimized_func2 = refined_hyperband(test_func2)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Hyperband and Bayesian Optimization')
plt.legend()
plt.show()