import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import expit
import random
import copy

class HyperbandBBO:
    def __init__(self, budget, dim, alpha=0.15, lambda_=0.01, mu=0.5):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.lambda_ = lambda_: lambda_
        self.mu = mu
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim
        self.func_evals = 0
        self.func_evals_evals = 0
        self.search_space = copy.deepcopy(self.search_space)
        self.search_space_dim = copy.deepcopy(self.search_space_dim)

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
            # Calculate the new fitness
            new_fitness = self.calculate_new_fitness(func_value)
            # Store the new fitness
            self.func_evals_evals = new_fitness
            # Refine the strategy
            self.refine_strategy(func_value, new_fitness)
        # Evaluate the function at the final point in the search space
        func_value = func(self.search_space)
        # Calculate the final fitness
        final_fitness = self.calculate_final_fitness(func_value)
        # Store the final fitness
        self.func_evals_evals = final_fitness
        return func_value

    def calculate_new_fitness(self, func_value):
        # Calculate the new fitness using the given algorithm
        new_fitness = np.exp(-((func_value - self.mu) / self.alpha)**2)
        return new_fitness

    def refine_strategy(self, func_value, new_fitness):
        # Refine the strategy based on the new fitness
        if new_fitness > self.func_evals_evals:
            self.search_space = (self.search_space[0] - 1, self.search_space[1] + 1)
        elif new_fitness < self.func_evals_evals:
            self.search_space = (self.search_space[0] + 1, self.search_space[1] - 1)

    def calculate_final_fitness(self, func_value):
        # Calculate the final fitness using the given algorithm
        final_fitness = np.exp(-((func_value - self.mu) / self.alpha)**2)
        return final_fitness

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