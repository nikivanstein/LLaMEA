import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

class HyperbandBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim
        self.search_space_min = np.min(self.search_space)
        self.search_space_max = np.max(self.search_space)
        self.search_space_mean = np.mean(self.search_space)

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Sample a new point in the search space using Gaussian distribution
            x = np.random.uniform(self.search_space_min, self.search_space_max, size=self.search_space_dim)
            # Evaluate the function at the new point
            func_value = func(x)
            # Store the function value and the new point
            self.func_evals += 1
            self.func_evals_evals = func_value
            # Store the new point in the search space
            self.search_space = (self.search_space_min, self.search_space_max)
            # Refine the strategy based on the current fitness value
            if np.random.rand() < 0.15:
                # Use a linear search in the upper half of the search space
                new_x = np.linspace(self.search_space_mean, self.search_space_max, 100)
                new_func_value = func(new_x)
                if new_func_value < func_value:
                    self.search_space = (self.search_space_mean, self.search_space_max)
            else:
                # Use a linear search in the lower half of the search space
                new_x = np.linspace(self.search_space_min, self.search_space_mean, 100)
                new_func_value = func(new_x)
                if new_func_value < func_value:
                    self.search_space = (self.search_space_min, self.search_space_mean)
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