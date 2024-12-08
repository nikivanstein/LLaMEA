import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class HyperbandBBO:
    def __init__(self, budget, dim, algorithm):
        self.budget = budget
        self.dim = dim
        self.algorithm = algorithm
        self.func_evals = 0
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim

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
        # Update the individual based on the algorithm
        if self.algorithm == 'Hyperband':
            self.update_individual(x, func_value)
        elif self.algorithm == 'Bayesian Optimization':
            self.update_individual(x, func_value, self.budget)
        return func_value

    def update_individual(self, x, func_value, budget):
        # Refine the strategy based on the probability 0.15
        if np.random.rand() < 0.15:
            # Use the current individual as the new point
            new_x = x
        else:
            # Use a new point sampled from the search space
            new_x = np.random.uniform(*self.search_space, size=self.search_space_dim)
        # Evaluate the function at the new point
        new_func_value = self.algorithm(func, new_x)
        # Update the individual with the new point and function value
        self.func_evals += 1
        self.func_evals_evals = new_func_value
        self.search_space = (min(self.search_space[0], new_x), max(self.search_space[1], new_x))
        # Update the individual's fitness
        self.func_evals = np.sum([self.func_evals_evals, func_value])

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

hyperband = HyperbandBBO(budget=100, dim=10, algorithm='Hyperband')
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