import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

class HyperbandBBO:
    def __init__(self, budget, dim, alpha=0.15, beta=0.1, lambda_1=1.0, lambda_2=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
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

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

hyperband = HyperbandBBO(budget=100, dim=10, alpha=0.15, beta=0.1, lambda_1=1.0, lambda_2=1.0)
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
    # Sample a new point in the search space using Gaussian distribution
    x = np.random.uniform(*individual['search_space'], size=individual['search_space_dim'])
    # Evaluate the function at the new point
    func_value = func(x)
    # Store the function value and the new point
    individual['func_value'] = func_value
    individual['new_point'] = x
    # Refine the search space
    if func_value < individual['func_value'] - 2 * individual['search_space_dim'] * individual['lambda_1'] * func_value:
        individual['search_space'] = (individual['search_space'][0] - 0.1, individual['search_space'][1] + 0.1)
    elif func_value > individual['func_value'] + 2 * individual['search_space_dim'] * individual['lambda_2'] * func_value:
        individual['search_space'] = (individual['search_space'][0] + 0.1, individual['search_space'][1] - 0.1)
    return individual

# Update the individual
optimized_func1['individual'] = refine_strategy(optimized_func1['individual'], test_func1, 100)
optimized_func2['individual'] = refine_strategy(optimized_func2['individual'], test_func2, 100)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1['individual']['func_value'], optimized_func2['individual']['func_value']], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Hyperband and Bayesian Optimization')
plt.legend()
plt.show()