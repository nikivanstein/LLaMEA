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
        self.algorithm = 'HyperbandBBO'

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

    def optimize(self, func, algorithm):
        if algorithm == 'HyperbandBBO':
            return self.__call__(func)
        elif algorithm == 'BayesianOptimization':
            return minimize(func, x0=np.array([1.0]), method='SLSQP', bounds=[(-5.0, 5.0), (-5.0, 5.0)], n_iter=1000)
        else:
            raise ValueError("Invalid algorithm. Supported algorithms are 'HyperbandBBO' and 'BayesianOptimization'.")

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

# Refine the strategy using Bayesian Optimization
def optimize_hyperband_bbo(budget, dim):
    algorithm = 'BayesianOptimization'
    return minimize(test_func1, x0=np.array([1.0]), method='SLSQP', bounds=[(-5.0, 5.0), (-5.0, 5.0)], n_iter=1000)

# Evaluate the function using Bayesian Optimization
optimized_func1_bayes = optimize_hyperband_bbo(budget=100, dim=10)
optimized_func2_bayes = optimize_hyperband_bbo(budget=100, dim=10)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1_bayes, optimized_func2_bayes], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Hyperband and Bayesian Optimization')
plt.legend()
plt.show()