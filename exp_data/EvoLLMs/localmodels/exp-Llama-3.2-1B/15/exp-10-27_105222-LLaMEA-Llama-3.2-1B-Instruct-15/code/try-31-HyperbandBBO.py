import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

class HyperbandBBO:
    def __init__(self, budget, dim, initial_strategy):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim
        self.search_space_init = initial_strategy
        self.func_evals = 0
        self.search_space = self.search_space_init

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

    def update_strategy(self, func, evaluations):
        # Calculate the probability of each individual line
        probabilities = np.array([norm.cdf(x, loc=func(x), scale=1) for x in evaluations])
        
        # Normalize the probabilities
        probabilities = probabilities / np.sum(probabilities)
        
        # Refine the strategy
        new_strategy = np.random.choice(self.dim, size=self.dim, p=probabilities)
        
        # Update the search space
        self.search_space_init = new_strategy
        
        return new_strategy

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

hyperband = HyperbandBBO(budget=100, dim=10, initial_strategy=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
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
new_strategy = hyperband.update_strategy(test_func1, [10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
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