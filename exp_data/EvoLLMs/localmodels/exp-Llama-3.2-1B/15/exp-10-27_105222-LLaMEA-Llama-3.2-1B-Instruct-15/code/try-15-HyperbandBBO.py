# Description: Efficient Black Box Optimization using Hyperband and Bayesian Optimization
# Code: 
# ```python
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

    def bayesian_optimization(self, func, num_iterations):
        # Initialize the algorithm with a random initial point
        self.initial_individual = np.random.uniform(-5.0, 5.0, self.search_space_dim)
        self.bayesian_optimizer = minimize(self.func, self.initial_individual, method='SLSQP', bounds=self.search_space, options={'maxiter': num_iterations})

        # Refine the strategy based on the probability 0.15
        self.bayesian_optimizer.x = self.initial_individual
        for _ in range(num_iterations):
            # Evaluate the function at the current point
            func_value = self.func(self.bayesian_optimizer.x)
            # Update the point using Bayesian optimization
            self.bayesian_optimizer.x = np.random.uniform(*self.search_space, size=self.search_space_dim)
            # Store the new point in the search space
            self.search_space = (min(self.search_space[0], self.bayesian_optimizer.x), max(self.search_space[1], self.bayesian_optimizer.x))

    def __call__(self, func):
        self.bayesian_optimizer = minimize(self.func, self.initial_individual, method='SLSQP', bounds=self.search_space, options={'maxiter': 1000})
        self.bayesian_optimization(func, 1000)

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