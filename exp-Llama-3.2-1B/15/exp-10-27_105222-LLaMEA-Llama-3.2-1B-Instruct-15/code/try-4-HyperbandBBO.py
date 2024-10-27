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

    def bayesian_optimization(self, func, num_samples, num_iterations):
        # Initialize the bayesian optimization algorithm
        bayesian = minimize(lambda x: -func(x), np.zeros(self.search_space_dim), method="SLSQP", bounds=self.search_space, options={"maxiter": num_iterations})
        
        # Refine the strategy using probability 0.15
        bayesian.x = bayesian.x * 0.85 + np.random.normal(0, 0.2, size=bayesian.x.shape)
        
        # Evaluate the function at the refined point
        func_value = func(bayesian.x)
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

# Bayesian optimization
num_samples = 10
num_iterations = 100
optimized_func1_bayesian = hyperband.bayesian_optimization(test_func1, num_samples, num_iterations)
optimized_func2_bayesian = hyperband.bayesian_optimization(test_func2, num_samples, num_iterations)