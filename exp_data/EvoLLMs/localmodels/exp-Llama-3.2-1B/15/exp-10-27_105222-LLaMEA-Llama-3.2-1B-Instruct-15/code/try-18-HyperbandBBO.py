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
        self.search_space_range = [self.search_space[0], self.search_space[1]]
        self.search_space_step = self.search_space_range / 10

    def __call__(self, func, initial_individual):
        while self.func_evals < self.budget:
            # Sample a new point in the search space using Gaussian distribution
            new_individual = initial_individual
            for _ in range(self.dim):
                new_individual = np.random.normal(0, 1, self.search_space_dim)
            # Evaluate the function at the new point
            func_value = func(new_individual)
            # Store the function value and the new point
            self.func_evals += 1
            self.func_evals_evals = func_value
            # Store the new point in the search space
            new_individual = (new_individual + self.search_space_step * np.random.normal(0, 1, self.search_space_dim)) % self.search_space_range
        # Evaluate the function at the final point in the search space
        func_value = func(new_individual)
        return func_value

def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

# Example usage:
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