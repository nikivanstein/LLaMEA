import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class AdaptiveHyperbandBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim
        self.func_evals = 0
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim
        self.search_space_best = (-5.0, 5.0)
        self.search_space_best_fitness = -inf
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
            # Store the function value and the final point
            self.func_evals_evals = func_value
            # Store the final point in the search space
            self.search_space_best = (min(self.search_space_best[0], self.search_space[0]), max(self.search_space_best[1], self.search_space[1]))
            self.search_space_best_fitness = func_value
        # Evaluate the function at the final point in the search space
        func_value = func(self.search_space_best)
        # Store the function value and the final point
        self.func_evals_evals = func_value
        # Store the final point in the search space
        self.search_space_best = (min(self.search_space_best[0], self.search_space[0]), max(self.search_space_best[1], self.search_space[1]))
        self.search_space_best_fitness = func_value
        # Refine the strategy using Bayesian optimization
        if self.func_evals_evals < self.search_space_best_fitness:
            self.algorithm = 'BayesianOptimization'
        return func(self.search_space_best)

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

adaptive_hyperband = AdaptiveHyperbandBBO(budget=100, dim=10)
optimized_func1 = adaptive_hyperband(test_func1)
optimized_func2 = adaptive_hyperband(test_func2)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Adaptive Hyperband-BBO with Bayesian Optimization')
plt.legend()
plt.show()