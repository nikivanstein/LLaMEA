import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random

class HyperbandBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim
        self.func_evals = 0
        self.search_space = self.search_space

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
        # Evaluate the function at the final point in the search space
        func_value = func(self.search_space)
        return func_value

    def update_strategy(self, individual, dim):
        # Refine the strategy using Bayesian Optimization
        # Generate a new point in the search space using Gaussian distribution
        x = np.random.uniform(*self.search_space, size=dim)
        # Evaluate the function at the new point
        func_value = individual(x)
        # Calculate the posterior distribution of the function value
        posterior = self.func_evals_evals / self.func_evals
        # Sample a new point in the search space using Gaussian distribution
        new_x = np.random.uniform(*self.search_space, size=dim)
        # Evaluate the function at the new point
        new_func_value = individual(new_x)
        # Calculate the posterior distribution of the function value
        posterior_new = new_func_value / posterior
        # Normalize the posterior distributions
        posterior_new /= posterior_new.sum()
        # Sample a new point in the search space using Gaussian distribution
        new_x = np.random.uniform(*self.search_space, size=dim)
        # Evaluate the function at the new point
        new_func_value = individual(new_x)
        # Calculate the posterior distribution of the function value
        posterior_new = new_func_value / posterior_new
        # Normalize the posterior distributions
        posterior_new /= posterior_new.sum()
        # Sample a new point in the search space using Gaussian distribution
        new_x = np.random.uniform(*self.search_space, size=dim)
        # Evaluate the function at the new point
        new_func_value = individual(new_x)
        # Calculate the posterior distribution of the function value
        posterior_new = new_func_value / posterior_new
        # Normalize the posterior distributions
        posterior_new /= posterior_new.sum()
        # Return the new point in the search space
        return new_x

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

# Update the strategy
def update_strategy(individual, dim):
    return hyperband.update_strategy(individual, dim)

# Plot the results with updated strategy
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.plot([update_strategy(individual1, dim), update_strategy(individual2, dim)], label=['Optimized Function 1', 'Optimized Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Hyperband and Bayesian Optimization')
plt.legend()
plt.show()