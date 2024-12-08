import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class HyperbandBBO:
    def __init__(self, budget, dim, alpha, beta, gamma):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim
        self.func_evals = 0
        self.search_space_dim_evals = 0
        self.search_space_dim_range = (-5.0, 5.0)
        self.search_space_range = (-5.0, 5.0)

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
            self.search_space_range = (min(self.search_space_range[0], x), max(self.search_space_range[1], x))
        # Evaluate the function at the final point in the search space
        func_value = func(self.search_space_range)
        # Update the search space based on the alpha, beta, and gamma values
        if self.alpha > 0:
            self.search_space_dim = np.random.uniform(self.search_space_dim_range[0], self.search_space_dim_range[1], size=self.dim)
        elif self.alpha == 0:
            self.search_space_dim = np.random.uniform(self.search_space_dim_range[0], self.search_space_dim_range[1], size=self.dim)
        else:
            self.search_space_dim = self.search_space_dim / (self.alpha * self.beta * self.gamma)
        # Update the search space range based on the new search space dimension
        self.search_space_range = (min(self.search_space_range[0], self.search_space_dim), max(self.search_space_range[1], self.search_space_dim))
        return func_value

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

hyperband = HyperbandBBO(budget=100, dim=10, alpha=0.1, beta=0.1, gamma=0.1)
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