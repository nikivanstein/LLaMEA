import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random

class HyperbandBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim

    def __call__(self, func, algorithm, dim):
        if algorithm == 'BayesianOptimization':
            # Bayesian Optimization
            self.func_evals = 0
            self.search_space = (-5.0, 5.0)
            self.search_space_dim = self.dim
            for _ in range(self.budget):
                # Sample a new point in the search space using Gaussian distribution
                x = np.random.uniform(*self.search_space, size=self.search_space_dim)
                # Evaluate the function at the new point
                func_value = func(x)
                # Store the function value and the new point
                self.func_evals += 1
                self.func_evals_evals = func_value
                # Store the new point in the search space
                self.search_space = (min(self.search_space[0], x), max(self.search_space[1], x))
        elif algorithm == 'Hyperband':
            # Hyperband
            self.func_evals = 0
            self.search_space = (-5.0, 5.0)
            self.search_space_dim = self.dim
            best_func_value = np.inf
            best_individual = None
            for _ in range(self.budget):
                # Sample a new point in the search space using Gaussian distribution
                x = np.random.uniform(*self.search_space, size=self.search_space_dim)
                # Evaluate the function at the new point
                func_value = func(x)
                # Update the best function value and individual
                if func_value < best_func_value:
                    best_func_value = func_value
                    best_individual = x
            # Store the best function value and individual
            self.func_evals += 1
            self.func_evals_evals = best_func_value
            self.search_space = (min(self.search_space[0], best_individual), max(self.search_space[1], best_individual))
        else:
            raise ValueError("Invalid algorithm. Please choose 'BayesianOptimization' or 'Hyperband'.")

    def evaluate_fitness(self, individual):
        # Evaluate the function at the individual
        func_value = func(individual)
        # Return the function value and the individual
        return func_value, individual

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

hyperband = HyperbandBBO(budget=100, dim=10)
optimized_func1 = hyperband(test_func1, 'BayesianOptimization', 10)
optimized_func2 = hyperband(test_func2, 'Hyperband', 10)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1[1], optimized_func2[1]], [optimized_func1[0], optimized_func2[0]], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Individual')
plt.title('Black Box Optimization using Hyperband and Bayesian Optimization')
plt.legend()
plt.show()