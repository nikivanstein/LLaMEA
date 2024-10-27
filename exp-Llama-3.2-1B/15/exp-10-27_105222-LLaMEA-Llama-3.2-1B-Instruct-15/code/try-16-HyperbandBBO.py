import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import lognorm

class HyperbandBBO:
    def __init__(self, budget, dim, learning_rate=0.01, decay_rate=0.99, alpha=0.1):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.func_evals = 0

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Sample a new point in the search space using Gaussian distribution
            x = np.random.uniform(*self.search_space, size=self.search_space_dim)
            # Evaluate the function at the new point
            func_value = func(x)
            # Store the function value and the new point
            self.func_evals += 1
            self.func_evals_evals = func_value

            # Refine the strategy using Bayesian optimization
            # Compute the posterior distribution of the search space
            posterior = norm.pdf(self.search_space, loc=np.mean(self.func_evals_evals), scale=self.func_evals_evals / self.func_evals)
            # Compute the expected value of the function at the new point
            expected_value = np.mean([func(x) for x in np.linspace(self.search_space[0], self.search_space[1], 100)])
            # Update the search space using the expected value
            self.search_space = (self.search_space[0] + self.alpha * (self.search_space[1] - self.search_space[0]) * (expected_value - self.func_evals_evals), self.search_space[1] - self.alpha * (self.search_space[1] - self.search_space[0]) * (expected_value - self.func_evals_evals))
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